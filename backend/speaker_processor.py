"""Reference-voice validation and preprocessing + SpeakerProfile abstraction.

F5-TTS conditions on the reference WAVEFORM (and its transcript) directly, so no separate
speaker encoder is invented here. An optional GE2E embedding (resemblyzer) can be computed
for EVALUATION only (speaker similarity), not for generation.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

import numpy as np
import soundfile as sf


class ReferenceAudioError(ValueError):
    pass


@dataclass
class ValidationReport:
    ok: bool
    duration_sec: float = 0.0
    sample_rate: int = 0
    channels: int = 0
    silence_ratio: float = 0.0
    clipping_pct: float = 0.0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class SpeakerProfile:
    name: str
    audio_path: str
    sample_rate: int
    duration_sec: float
    ref_text: str = ""            # transcript of the reference clip (F5-TTS uses it)
    embedding_path: Optional[str] = None  # optional GE2E embedding (evaluation only)

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        p = directory / f"{self.name}_profile.json"
        p.write_text(json.dumps(asdict(self), indent=2))
        return p

    @classmethod
    def load(cls, path: Path) -> "SpeakerProfile":
        return cls(**json.loads(Path(path).read_text()))


def load_audio(path) -> tuple[np.ndarray, int]:
    """Load any common format to float32 (samples, channels). Falls back to FFmpeg for MP3 etc."""
    path = Path(path)
    try:
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        return data, sr
    except Exception:
        pass
    try:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "t.wav"
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(path), str(out)],
                           check=True, capture_output=True)
            data, sr = sf.read(str(out), dtype="float32", always_2d=True)
            return data, sr
    except FileNotFoundError as e:
        raise ReferenceAudioError("FFmpeg is not installed/on PATH; cannot decode this audio format. Use a WAV file or install FFmpeg.") from e
    except Exception as e:
        raise ReferenceAudioError(f"Could not decode audio file: {path.name}") from e


def _silence_ratio(mono: np.ndarray, sr: int, thresh_db: float = -40.0) -> float:
    frame = max(1, int(0.02 * sr))
    n = len(mono) // frame
    if n == 0:
        return 1.0
    fr = mono[: n * frame].reshape(n, frame)
    rms = np.sqrt((fr ** 2).mean(axis=1) + 1e-12)
    return float((20 * np.log10(rms + 1e-12) < thresh_db).mean())


def validate_reference_audio(path, min_sec: float = 6.0, max_sec: float = 10.0) -> ValidationReport:
    path = Path(path)
    if not path.exists():
        return ValidationReport(False, errors=[f"File not found: {path}"])
    if path.suffix.lower() not in {".wav", ".mp3", ".flac", ".ogg", ".m4a"}:
        return ValidationReport(False, errors=[f"Unsupported file type '{path.suffix}'. Use WAV or MP3."])
    try:
        data, sr = load_audio(path)
    except ReferenceAudioError as e:
        return ValidationReport(False, errors=[str(e)])
    mono = data.mean(axis=1)
    dur = len(mono) / sr
    rep = ValidationReport(True, duration_sec=dur, sample_rate=sr, channels=data.shape[1])
    rep.silence_ratio = _silence_ratio(mono, sr)
    rep.clipping_pct = float((np.abs(data) >= 0.999).mean() * 100)
    if dur < min_sec:
        rep.errors.append(f"Too short ({dur:.1f}s). Need {min_sec:.0f}-{max_sec:.0f}s.")
    if dur > max_sec + 5:
        rep.errors.append(f"Too long ({dur:.1f}s). Need {min_sec:.0f}-{max_sec:.0f}s.")
    elif dur > max_sec:
        rep.warnings.append(f"Longer than {max_sec:.0f}s ({dur:.1f}s); it will be trimmed.")
    if rep.silence_ratio > 0.6:
        rep.errors.append(f"Mostly silence ({rep.silence_ratio:.0%}). Record closer to the microphone.")
    elif rep.silence_ratio > 0.35:
        rep.warnings.append(f"High silence ratio ({rep.silence_ratio:.0%}).")
    if rep.clipping_pct > 1.0:
        rep.errors.append(f"Heavy clipping ({rep.clipping_pct:.1f}% of samples). Re-record at lower gain.")
    elif rep.clipping_pct > 0.1:
        rep.warnings.append(f"Some clipping ({rep.clipping_pct:.2f}%).")
    if sr < 16000:
        rep.warnings.append(f"Low sample rate ({sr} Hz); quality may suffer.")
    rep.ok = not rep.errors
    return rep


def trim_silence(mono: np.ndarray, sr: int, thresh_db: float = -40.0, keep_ms: int = 100) -> np.ndarray:
    frame = max(1, int(0.01 * sr))
    n = len(mono) // frame
    if n == 0:
        return mono
    rms = np.sqrt((mono[: n * frame].reshape(n, frame) ** 2).mean(axis=1) + 1e-12)
    active = np.where(20 * np.log10(rms + 1e-12) > thresh_db)[0]
    if len(active) == 0:
        return mono
    keep = int(keep_ms / 1000 * sr)
    s = max(0, active[0] * frame - keep)
    e = min(len(mono), (active[-1] + 1) * frame + keep)
    return mono[s:e]


def preprocess_reference_audio(path, out_dir, name: str = "speaker", target_sr: int = 24000,
                               max_sec: float = 10.0, ref_text: str = "") -> SpeakerProfile:
    """mono -> resample -> trim silence -> peak-normalize -> cap length -> save WAV + profile."""
    import librosa
    rep = validate_reference_audio(path, max_sec=max_sec)
    if not rep.ok:
        raise ReferenceAudioError("; ".join(rep.errors))
    data, sr = load_audio(path)
    mono = data.mean(axis=1)
    if sr != target_sr:
        mono = librosa.resample(mono, orig_sr=sr, target_sr=target_sr)
    mono = trim_silence(mono, target_sr)
    mono = mono[: int(max_sec * target_sr)]
    peak = float(np.abs(mono).max())
    if peak > 0:
        mono = mono * (0.95 / peak)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{name}_ref.wav"
    sf.write(str(out), mono.astype(np.float32), target_sr, subtype="PCM_16")
    prof = SpeakerProfile(name=name, audio_path=str(out), sample_rate=target_sr,
                          duration_sec=len(mono) / target_sr, ref_text=ref_text)
    prof.save(out_dir)
    return prof


def compute_speaker_embedding(audio_path) -> np.ndarray:
    """GE2E (Resemblyzer) d-vector, used only for evaluation. Raises a clear message if absent."""
    try:
        from resemblyzer import VoiceEncoder, preprocess_wav
    except ImportError as e:
        raise ImportError("Speaker-similarity evaluation needs: pip install resemblyzer") from e
    return VoiceEncoder().embed_utterance(preprocess_wav(str(audio_path)))
