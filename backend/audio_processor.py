"""Audio utilities: chunk inspection, trimming, silence, safe concatenation, loudness, export."""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import soundfile as sf


class AudioProcessingError(RuntimeError):
    pass


@dataclass
class ChunkInspection:
    ok: bool
    duration_sec: float = 0.0
    rms_db: float = -120.0
    clipping_pct: float = 0.0
    problems: tuple = ()


def to_mono(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x if x.ndim == 1 else x.mean(axis=1 if x.shape[0] > x.shape[1] else 0)


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x
    import librosa
    return librosa.resample(x, orig_sr=sr_in, target_sr=sr_out)


def inspect_chunk(x: np.ndarray, sr: int, min_sec: float = 0.2) -> ChunkInspection:
    """Detect corrupted/empty/NaN/too-quiet/clipped chunks."""
    if x is None or len(x) == 0:
        return ChunkInspection(False, problems=("empty",))
    x = to_mono(x)
    if not np.isfinite(x).all():
        return ChunkInspection(False, problems=("nan_or_inf",))
    dur = len(x) / sr
    rms = float(np.sqrt((x ** 2).mean()))
    rms_db = 20 * np.log10(rms + 1e-9)
    clip = float((np.abs(x) >= 0.999).mean() * 100)
    problems = []
    if dur < min_sec: problems.append("too_short")
    if rms_db < -60: problems.append("silent")
    if clip > 1.0: problems.append("clipped")
    fatal = {"too_short", "silent"}
    return ChunkInspection(not (fatal & set(problems)), dur, rms_db, clip, tuple(problems))


def trim_edges(x: np.ndarray, sr: int, thresh_db: float = -45.0, keep_ms: int = 60) -> np.ndarray:
    frame = max(1, int(0.01 * sr)); n = len(x) // frame
    if n == 0: return x
    rms = np.sqrt((x[: n * frame].reshape(n, frame) ** 2).mean(axis=1) + 1e-12)
    act = np.where(20 * np.log10(rms + 1e-12) > thresh_db)[0]
    if len(act) == 0: return x
    k = int(keep_ms / 1000 * sr)
    return x[max(0, act[0] * frame - k): min(len(x), (act[-1] + 1) * frame + k)]


def make_silence(ms: float, sr: int) -> np.ndarray:
    return np.zeros(int(sr * ms / 1000), dtype=np.float32)


def fade(x: np.ndarray, sr: int, ms: float = 10) -> np.ndarray:
    n = min(int(sr * ms / 1000), len(x) // 2)
    if n <= 0: return x
    x = x.copy(); ramp = np.linspace(0, 1, n, dtype=np.float32)
    x[:n] *= ramp; x[-n:] *= ramp[::-1]
    return x


def concatenate(segments: List[np.ndarray], srs: List[int], pauses_ms: List[float],
                target_sr: int = 24000) -> np.ndarray:
    """Concatenate mono-ised, resampled segments. pauses_ms[i] is inserted AFTER segment i
    (len(pauses_ms) may be len(segments)-1 or len(segments); extra trailing pause ignored)."""
    if len(segments) != len(srs):
        raise AudioProcessingError("segments and sample-rates length mismatch")
    out: List[np.ndarray] = []
    for i, (seg, sr) in enumerate(zip(segments, srs)):
        seg = resample(to_mono(seg), sr, target_sr)
        out.append(fade(seg, target_sr))
        if i < len(segments) - 1:
            out.append(make_silence(pauses_ms[i] if i < len(pauses_ms) else 0, target_sr))
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def measure_loudness(x: np.ndarray, sr: int) -> Optional[float]:
    """Integrated LUFS via pyloudnorm if installed; otherwise RMS dBFS (documented fallback)."""
    try:
        import pyloudnorm as pyln
        return float(pyln.Meter(sr).integrated_loudness(x))
    except Exception:
        return float(20 * np.log10(np.sqrt((x ** 2).mean()) + 1e-9))


def normalize_loudness(x: np.ndarray, sr: int, target: float = -18.0, peak_limit: float = 0.97) -> np.ndarray:
    """Gain to target loudness (LUFS if pyloudnorm present, else RMS dBFS), then peak-limit."""
    if len(x) == 0: return x
    cur = measure_loudness(x, sr)
    if cur is None or cur < -70: return x
    y = x * (10 ** ((target - cur) / 20))
    peak = float(np.abs(y).max())
    if peak > peak_limit:
        y = y * (peak_limit / peak)  # simple peak scaling avoids hard clipping
    return y.astype(np.float32)


def save_wav(x: np.ndarray, sr: int, path) -> Path:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(x, -1, 1), sr, subtype="PCM_16")
    return path


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def wav_to_mp3(wav_path, mp3_path, bitrate: str = "192k") -> Path:
    if not ffmpeg_available():
        raise AudioProcessingError("FFmpeg not found on PATH - MP3 export unavailable. Install FFmpeg (WAV export still works).")
    mp3_path = Path(mp3_path); mp3_path.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav_path),
                        "-codec:a", "libmp3lame", "-b:a", bitrate, str(mp3_path)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise AudioProcessingError(f"FFmpeg MP3 conversion failed: {r.stderr.strip()[:200]}")
    return mp3_path
