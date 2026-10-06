from __future__ import annotations

import importlib
import inspect
from pathlib import Path

from backend.tts_engine import BaseTTSEngine, TTSGenerationError, TTSUnavailableError


class F5TTSEngine(BaseTTSEngine):
    """Lazy, dependency-aware wrapper around the real F5-TTS package when installed."""

    name = "f5-tts"

    def is_available(self) -> bool:
        try:
            self._resolve_model_cls()
        except TTSUnavailableError:
            return False
        return True

    @staticmethod
    def _resolve_model_cls():
        try:
            module = importlib.import_module("f5_tts.api")
        except Exception as exc:
            raise TTSUnavailableError(
                "F5-TTS is unavailable or its API dependencies could not be imported. "
                "Install f5-tts and its runtime dependencies in the active environment."
            ) from exc

        model_cls = getattr(module, "F5TTS", None)
        if not inspect.isclass(model_cls):
            raise TTSUnavailableError(
                "The installed F5-TTS package does not expose the expected f5_tts.api.F5TTS API."
            )
        return model_cls

    def load_model(self):
        try:
            cls = self._resolve_model_cls()
            return cls(device=self.device, hf_cache_dir=str(self.model_dir))
        except TTSUnavailableError:
            raise
        except Exception as exc:  # pragma: no cover - model/download specific
            raise TTSUnavailableError(
                f"F5-TTS could not initialize its pretrained model: {exc}"
            ) from exc

    def generate(self, text: str, reference_audio: str | Path, output_path: str | Path, **kwargs) -> Path:
        if not text or not text.strip():
            raise TTSGenerationError("Text chunk is empty; TTS generation requires non-empty text.")

        ref_text = kwargs.get("ref_text")
        if not isinstance(ref_text, str) or not ref_text.strip():
            raise TTSGenerationError(
                "Reference transcript is required for F5-TTS. Enter the exact words spoken in the reference audio; "
                "the transcript is not inferred automatically."
            )

        if not Path(reference_audio).exists():
            raise TTSGenerationError(f"Reference audio not found: {reference_audio}")

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        try:
            model = self.ensure_ready()
            # The current F5-TTS API returns (waveform, sample_rate, spectrogram)
            # and writes the requested WAV as a side effect when file_wave is set.
            model.infer(
                ref_file=str(reference_audio),
                ref_text=ref_text.strip(),
                gen_text=text,
                speed=kwargs.get("speed", 1.0),
                cfg_strength=kwargs.get("cfg_strength", 2.0),
                file_wave=str(output),
            )
        except TTSUnavailableError:
            raise
        except Exception as exc:
            raise TTSGenerationError(f"F5-TTS inference failed: {exc}") from exc

        if output.is_file():
            return output

        raise TTSGenerationError(
            "F5-TTS did not create an output file. Check the model installation, model cache, and output path."
        )
