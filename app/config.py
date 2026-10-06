"""Central configuration. All paths/settings come from environment variables
(optionally loaded from .env) - nothing is hard-coded."""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # python-dotenv optional at runtime
    pass

ROOT = Path(__file__).resolve().parent.parent


def _path(var: str, default: str) -> Path:
    raw = os.getenv(var) or default
    p = Path(raw)
    return p if p.is_absolute() else (ROOT / p).resolve()


@dataclass
class Settings:
    model_cache_dir: Path = field(default_factory=lambda: _path("MODEL_CACHE_DIR", "models"))
    input_dir: Path = field(default_factory=lambda: _path("INPUT_DIR", "data/input"))
    reference_dir: Path = field(default_factory=lambda: _path("REFERENCE_DIR", "data/reference_voice"))
    processed_dir: Path = field(default_factory=lambda: _path("PROCESSED_DIR", "data/processed"))
    output_dir: Path = field(default_factory=lambda: _path("OUTPUT_DIR", "data/output"))
    device: str = field(default_factory=lambda: os.getenv("DEVICE", "auto").split("#")[0].strip().lower())
    default_tts_model: str = field(default_factory=lambda: os.getenv("DEFAULT_TTS_MODEL", "f5tts").split("#")[0].strip().lower())

    # Reference-audio rules
    ref_min_sec: float = 6.0
    ref_max_sec: float = 10.0
    ref_target_sr: int = 24000       # F5-TTS operates at 24 kHz

    # Chunking defaults
    max_chunk_chars: int = 400
    context_window_sentences: int = 2
    pause_sentence_ms: int = 350
    pause_paragraph_ms: int = 800
    pause_chapter_ms: int = 1500
    target_lufs: float = -18.0

    def ensure_dirs(self) -> None:
        for p in (self.model_cache_dir, self.input_dir, self.reference_dir,
                  self.processed_dir, self.output_dir):
            p.mkdir(parents=True, exist_ok=True)


def resolve_device(requested: str = "auto") -> str:
    """Return 'cuda' or 'cpu'. Never raises if torch/CUDA are missing."""
    if requested in ("cpu", "cuda"):
        if requested == "cuda":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return "cpu"
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s
