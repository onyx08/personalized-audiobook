from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import soundfile as sf

from backend.audio_processor import concatenate, normalize_loudness, save_wav
from backend.context_manager import LongFormContextManager
from backend.speaker_processor import preprocess_reference_audio
from backend.text_processor import Chapter, detect_chapters, extract_text
from backend.tts_engine import TTSUnavailableError, detect_tts_backend


@dataclass
class PipelineResult:
    output_dir: Path
    chapter_count: int = 0
    chunk_count: int = 0
    generated_files: List[str] = field(default_factory=list)
    status: str = "pending"
    warnings: List[str] = field(default_factory=list)


def _slugify(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower())
    return value.strip("-") or "chapter"


def assemble_chapter_audio(audio_segments: Iterable[bytes | np.ndarray | str | Path], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    segments = list(audio_segments)
    if not segments:
        path.write_bytes(b"")
        return path

    values = []
    for seg in segments:
        if isinstance(seg, (bytes, bytearray)):
            values.append(seg)
        elif isinstance(seg, (str, Path)):
            p = Path(seg)
            if p.exists():
                try:
                    data, _ = sf.read(str(p), dtype="float32", always_2d=False)
                    values.append(np.asarray(data, dtype=np.float32))
                except Exception:
                    values.append(np.asarray([], dtype=np.float32))
            else:
                values.append(np.asarray([], dtype=np.float32))
        else:
            arr = np.asarray(seg, dtype=np.float32)
            values.append(arr)

    if all(isinstance(v, (bytes, bytearray)) for v in values):
        payload = b"".join(values)
        path.write_bytes(payload)
        return path

    arrs = [np.asarray(v, dtype=np.float32) for v in values if np.asarray(v).size > 0]
    if not arrs:
        path.write_bytes(b"")
        return path
    combined = np.concatenate(arrs)
    if combined.ndim == 2:
        combined = combined.mean(axis=1)
    save_wav(normalize_loudness(combined, 24000), 24000, path)
    return path


class BookPipeline:
    def __init__(
        self,
        book_path: str | Path,
        reference_audio: str | Path,
        output_dir: str | Path,
        ref_text: str,
        engine=None,
        max_chunk_chars: int = 250,
    ):
        self.book_path = Path(book_path)
        self.reference_audio = Path(reference_audio)
        self.output_dir = Path(output_dir)
        self.ref_text = ref_text.strip() if isinstance(ref_text, str) else ""
        if not self.ref_text:
            raise ValueError(
                "Reference transcript is required. Enter the exact words spoken in the reference audio; "
                "it is not inferred automatically."
            )
        self.engine = engine or detect_tts_backend()
        self.max_chunk_chars = max_chunk_chars

    def _prepare_reference(self) -> Path:
        if not self.reference_audio.exists():
            raise FileNotFoundError(f"Reference audio not found: {self.reference_audio}")
        processed_dir = self.output_dir / "reference"
        profile = preprocess_reference_audio(
            self.reference_audio,
            processed_dir,
            name="speaker",
            target_sr=24000,
            max_sec=10.0,
            ref_text=self.ref_text,
        )
        return Path(profile.audio_path)

    def run(self) -> PipelineResult:
        if not self.book_path.exists():
            raise FileNotFoundError(f"Book input not found: {self.book_path}")

        # Ensure the output directory exists before preparing a new run.
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Start each generation with a clean set of generated audio artifacts.
        # Only files produced by this pipeline are removed.
        chunks_dir = self.output_dir / "chunks"
        chunks_dir.mkdir(parents=True, exist_ok=True)

        for old_chunk in chunks_dir.glob("chapter_*_chunk_*.wav"):
            old_chunk.unlink()

        for old_chapter in self.output_dir.glob("chapter_*.wav"):
            old_chapter.unlink()

        final_path = self.output_dir / "audiobook_complete.wav"
        if final_path.exists():
            final_path.unlink()

        text = extract_text(self.book_path)
        chapters = detect_chapters(text)

        context = LongFormContextManager(
            max_chunk_chars=self.max_chunk_chars
        )
        prepared = context.prepare_book(chapters)

        if self.engine is None or not self.engine.is_available():
            raise TTSUnavailableError(
                "No usable TTS backend is available. "
                "The project is configured to use XTTS-v2 for "
                "reference-voice conditioned narration."
            )

        reference_path = self._prepare_reference()

        result = PipelineResult(
            output_dir=self.output_dir,
            chapter_count=len(chapters),
            status="running",
        )

        complete_paths: List[Path] = []

        for chapter in chapters:
            chunks = prepared.get(chapter.chapter_id, [])
            chapter_data = []

            chapter_path = (
                self.output_dir
                / f"chapter_{chapter.chapter_id:02d}_{_slugify(chapter.title)}.wav"
            )

            for chunk in chunks:
                chunk_path = (
                    chunks_dir
                    / f"chapter_{chapter.chapter_id:02d}_chunk_{chunk.chunk_id:03d}.wav"
                )

                generated = self.engine.generate(
                    chunk.text,
                    reference_path,
                    chunk_path,
                    ref_text=self.ref_text,
                    speed=chunk.pace,
                    emotion=chunk.emotion,
                )

                if generated.exists():
                    chapter_data.append(generated)
                    result.generated_files.append(str(generated))
                else:
                    raise FileNotFoundError(
                        f"Expected generated chunk file was not created: {chunk_path}"
                    )

            assembled = assemble_chapter_audio(
                chapter_data,
                chapter_path,
            )

            complete_paths.append(assembled)
            result.chunk_count += len(chunks)

        assemble_chapter_audio(
            complete_paths,
            final_path,
        )

        result.generated_files.append(str(final_path))
        result.status = "complete"

        return result