from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from backend.audio_processor import measure_loudness
from backend.speaker_processor import compute_speaker_embedding


@dataclass
class EvaluationResult:
    audio_path: Optional[str] = None
    reference_audio: Optional[str] = None
    speaker_similarity: Optional[float] = None
    duration_sec: Optional[float] = None
    chunk_count: int = 0
    processing_time_sec: float = 0.0
    quality_checks: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)


def evaluate_audio(reference_audio: str | Path, generated_audio: str | Path, chunk_count: int = 0) -> EvaluationResult:
    start = time.time()
    result = EvaluationResult(
        audio_path=str(generated_audio),
        reference_audio=str(reference_audio),
        chunk_count=chunk_count,
    )

    try:
        ref_emb = compute_speaker_embedding(str(reference_audio))
        gen_emb = compute_speaker_embedding(str(generated_audio))
        if ref_emb is not None and gen_emb is not None:
            result.speaker_similarity = float((ref_emb * gen_emb).sum() / (abs(ref_emb).sum() * abs(gen_emb).sum() + 1e-9))
    except Exception as exc:
        result.notes.append(f"Speaker similarity unavailable: {exc}")

    try:
        data, sr = __import__("soundfile").read(str(generated_audio), always_2d=False)
        result.duration_sec = float(len(data) / sr)
    except Exception as exc:
        result.notes.append(f"Duration unavailable: {exc}")

    try:
        import numpy as np
        data, sr = __import__("soundfile").read(str(generated_audio), dtype="float32", always_2d=False)
        result.quality_checks["rms_dbfs"] = float(measure_loudness(np.asarray(data, dtype=np.float32), sr))
    except Exception as exc:
        result.notes.append(f"Audio check unavailable: {exc}")

    result.processing_time_sec = time.time() - start
    return result
