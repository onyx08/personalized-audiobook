from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

from app.config import get_settings, resolve_device


class TTSUnavailableError(RuntimeError):
    """Raised when the configured TTS backend or its dependencies are missing."""


class TTSGenerationError(RuntimeError):
    """Raised when TTS generation fails even though the backend is installed."""


class BaseTTSEngine(ABC):
    """Abstract TTS API for pluggable speech-generation backends."""

    name = "base"

    def __init__(self, model_dir: Optional[str | Path] = None, device: Optional[str] = None):
        settings = get_settings()
        base_dir = Path(model_dir) if model_dir is not None else settings.model_cache_dir
        self.model_dir = (
            base_dir
            if base_dir.is_absolute()
            else (settings.model_cache_dir / base_dir).resolve()
        )
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.device = resolve_device(device or settings.device)
        self._model: Any = None

    @abstractmethod
    def is_available(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def load_model(self):
        raise NotImplementedError

    @abstractmethod
    def generate(
        self,
        text: str,
        reference_audio: str | Path,
        output_path: str | Path,
        **kwargs,
    ) -> Path:
        raise NotImplementedError

    def ensure_ready(self):
        if self._model is None:
            self._model = self.load_model()
        return self._model

    @staticmethod
    def _coerce_path(path: str | Path) -> Path:
        return Path(path)

    def describe(self) -> str:
        return f"{self.name} on {self.device}"


def detect_tts_backend():
    """Return the first available TTS backend without failing on import errors."""

    # XTTS-v2 is the primary backend for the working voice-cloning environment.
    try:
        from backend.xtts_engine import XTTSEngine

        backend = XTTSEngine()
        if backend.is_available():
            return backend
    except Exception:
        pass

    # Keep the existing F5-TTS backend as a fallback.
    try:
        from backend.f5_tts_engine import F5TTSEngine

        backend = F5TTSEngine()
        if backend.is_available():
            return backend
    except Exception:
        pass

    return None


def __getattr__(name: str):
    """Keep historical backend import paths without eager imports."""

    if name == "XTTSEngine":
        from backend.xtts_engine import XTTSEngine

        return XTTSEngine

    if name == "F5TTSEngine":
        from backend.f5_tts_engine import F5TTSEngine

        return F5TTSEngine

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
