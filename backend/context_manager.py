"""Long-Form Context Manager - orchestration-level continuity for audiobook narration.

WHAT IS ACTUAL MODEL CONDITIONING vs ORCHESTRATION (be honest in the viva):
  * Model conditioning (depends on the TTS engine API):
      - the same reference audio + transcript for every chunk (speaker identity)
      - per-chunk `speed` value (from `pace`)
      - fixed seed / sampling settings
  * Orchestration only (done here, NOT inside the network):
      - paragraph-aware chunking at punctuation boundaries
      - rolling textual context (`previous_context`) - stored for logging/diagnostics and
        passed to engines ONLY if their API supports it (F5-TTS public API does not)
      - rule-based emotion/energy/pace estimates, smoothed across neighbours
      - adaptive pauses between chunks
This module does NOT alter any neural network's hidden state.
Emotion/energy/pace are punctuation/lexical HEURISTICS, not learned predictions.
"""
from __future__ import annotations

import json
import re
from collections import deque
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Deque, Dict, List, Optional

from backend.text_processor import (Chapter, normalize_text, segment_paragraphs,
                                    segment_sentences, _split_long)


@dataclass
class ChunkMeta:
    chunk_id: int                 # global, monotonically increasing
    chapter_id: int
    paragraph_id: int
    sentence_id: int              # index of the first sentence in the paragraph
    text: str
    previous_context: str
    emotion: str                  # neutral | excited | questioning | somber | tense
    pace: float                   # speed multiplier (1.0 = normal)
    energy: float                 # 0..1 relative loudness/intensity estimate
    pause_before: int             # ms
    pause_after: int              # ms
    is_dialogue: bool = False
    is_paragraph_start: bool = False
    is_paragraph_end: bool = False
    status: str = "pending"       # pending | done | failed
    audio_path: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


_SOMBER = re.compile(r"\b(died|death|dead|funeral|grief|tears|wept|sorrow|alone|lonely|lost|silence)\b", re.I)
_TENSE = re.compile(r"\b(suddenly|screamed|ran|danger|blood|fear|afraid|hurry|shouted|trembl\w+)\b", re.I)


class LongFormContextManager:
    def __init__(self, max_chunk_chars: int = 250, context_window: int = 2,
                 pause_sentence_ms: int = 350, pause_paragraph_ms: int = 800,
                 pause_chapter_ms: int = 1500, base_pace: float = 1.0,
                 smoothing: float = 0.5, context_chars: int = 300):
        if not 0 <= smoothing < 1:
            raise ValueError("smoothing must be in [0, 1)")
        self.max_chunk_chars = max_chunk_chars
        self.context_window = max(0, context_window)
        self.pause_sentence_ms = pause_sentence_ms
        self.pause_paragraph_ms = pause_paragraph_ms
        self.pause_chapter_ms = pause_chapter_ms
        self.base_pace = base_pace
        self.smoothing = smoothing
        self.context_chars = context_chars
        self._next_id = 0

    # ---------------------------------------------------------------- segmentation
    def segment_chapter(self, chapter: Chapter) -> List[List[str]]:
        """chapter -> paragraphs -> sentences (nested list)."""
        text = normalize_text(chapter.text)
        return [segment_sentences(p) for p in segment_paragraphs(text)]

    def _group(self, sentences: List[str]) -> List[tuple]:
        """Greedy grouping of sentences into chunks; returns [(first_sentence_idx, text)].
        Never merges across paragraphs. Over-long sentences are split at clause punctuation."""
        units = []
        for i, s in enumerate(sentences):
            parts = _split_long(s, self.max_chunk_chars) if len(s) > self.max_chunk_chars else [s]
            units.extend((i, p) for p in parts)
        groups, cur_idx, cur = [], None, ""
        for idx, text in units:
            if cur and len(cur) + 1 + len(text) > self.max_chunk_chars:
                groups.append((cur_idx, cur)); cur_idx, cur = idx, text
            else:
                if cur_idx is None: cur_idx = idx
                cur = f"{cur} {text}".strip()
        if cur:
            groups.append((cur_idx, cur))
        return groups

    # ---------------------------------------------------------------- style heuristics
    @staticmethod
    def estimate_style(text: str) -> Dict[str, object]:
        dialogue = text.count('"') >= 2 or text.lstrip().startswith('"')
        emotion, energy, pace = "neutral", 0.5, 1.0
        if "!" in text:
            emotion, energy, pace = "excited", 0.8, 1.03
        elif text.rstrip().endswith("?") or text.rstrip().endswith('?"'):
            emotion, energy = "questioning", 0.6
        elif _TENSE.search(text):
            emotion, energy, pace = "tense", 0.7, 1.04
        elif _SOMBER.search(text) or "..." in text:
            emotion, energy, pace = "somber", 0.3, 0.93
        if dialogue:
            energy = min(1.0, energy + 0.1)
        return {"emotion": emotion, "energy": energy, "pace": pace, "is_dialogue": dialogue}

    def _smooth(self, prev: Optional[ChunkMeta], energy: float, pace: float) -> tuple:
        if prev is None or self.smoothing == 0:
            return energy, pace
        a = self.smoothing
        return (round(a * prev.energy + (1 - a) * energy, 3),
                round(a * prev.pace + (1 - a) * pace, 3))

    def _pauses(self, text: str, para_start: bool, para_end: bool, chap_start: bool, prev: Optional[ChunkMeta]) -> tuple:
        tail = text.rstrip('"\') ')[-1:]
        after = self.pause_sentence_ms
        if tail in ",;:": after = int(self.pause_sentence_ms * 0.45)
        elif tail == "?": after = int(self.pause_sentence_ms * 1.15)
        elif tail == "!": after = int(self.pause_sentence_ms * 1.05)
        if text.rstrip().endswith("...") or text.endswith("\u2014"): after = int(self.pause_sentence_ms * 1.3)
        if para_end: after = self.pause_paragraph_ms
        before = 0
        if chap_start: before = self.pause_chapter_ms
        elif para_start and prev is not None: before = 0  # paragraph gap comes from previous pause_after
        return before, after

    # ---------------------------------------------------------------- main API
    def prepare_chapter(self, chapter: Chapter) -> List[ChunkMeta]:
        """Produce ordered, fully-annotated chunks for one chapter. Context resets at chapter start
        (a new chapter is a topical break) but carries across paragraphs inside the chapter."""
        window: Deque[str] = deque(maxlen=self.context_window or 1)
        chunks: List[ChunkMeta] = []
        prev: Optional[ChunkMeta] = None
        for p_id, sentences in enumerate(self.segment_chapter(chapter)):
            groups = self._group(sentences)
            for g_i, (s_id, text) in enumerate(groups):
                style = self.estimate_style(text)
                energy, pace = self._smooth(prev, style["energy"], style["pace"] * self.base_pace)
                para_start, para_end = g_i == 0, g_i == len(groups) - 1
                before, after = self._pauses(text, para_start, para_end, not chunks, prev)
                ctx = " ".join(window)[-self.context_chars:] if self.context_window else ""
                meta = ChunkMeta(self._next_id, chapter.chapter_id, p_id, s_id, text, ctx,
                                 style["emotion"], round(pace, 3), energy, before, after,
                                 style["is_dialogue"], para_start, para_end)
                self._next_id += 1
                chunks.append(meta); prev = meta
                window.extend(segment_sentences(text, use_spacy=False))
        return chunks

    def prepare_book(self, chapters: List[Chapter]) -> Dict[int, List[ChunkMeta]]:
        return {c.chapter_id: self.prepare_chapter(c) for c in chapters}

    # ---------------------------------------------------------------- persistence
    @staticmethod
    def save_metadata(chunks: List[ChunkMeta], path) -> Path:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([c.to_dict() for c in chunks], indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    @staticmethod
    def load_metadata(path) -> List[ChunkMeta]:
        return [ChunkMeta(**d) for d in json.loads(Path(path).read_text(encoding="utf-8"))]
