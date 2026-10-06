from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from backend.tts_engine import BaseTTSEngine, TTSGenerationError


class XTTSEngine(BaseTTSEngine):
    """XTTS-v2 zero-shot voice-cloning backend."""

    name = "xtts-v2"

    def __init__(
        self,
        model_dir: Optional[str | Path] = None,
        device: Optional[str] = None,
    ):
        super().__init__(model_dir=model_dir, device=device)
        self._tts = None

    def is_available(self) -> bool:
        try:
            from TTS.api import TTS  # noqa: F401
            return True
        except Exception:
            return False

    def load_model(self):
        # Required on Windows so TorchCodec can find FFmpeg and PyTorch DLLs.
        ffmpeg_bin = Path(r"C:\ffmpeg\bin")
        torch_lib = (
            Path.cwd()
            / ".venv-xtts"
            / "Lib"
            / "site-packages"
            / "torch"
            / "lib"
        )

        if ffmpeg_bin.exists():
            os.add_dll_directory(str(ffmpeg_bin))
        if torch_lib.exists():
            os.add_dll_directory(str(torch_lib))

        from TTS.api import TTS

        model = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
        model.to("cpu")
        return model

    def generate(
        self,
        text: str,
        reference_audio: str | Path,
        output_path: str | Path,
        **kwargs,
    ) -> Path:
        if not text or not text.strip():
            raise ValueError("TTS text cannot be empty.")

        reference_audio = Path(reference_audio)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not reference_audio.exists():
            raise FileNotFoundError(
                f"Reference audio not found: {reference_audio}"
            )

        try:
            tts = self.ensure_ready()

            tts.tts_to_file(
                text=text,
                speaker_wav=str(reference_audio),
                language=kwargs.get("language", "en"),
                file_path=str(output_path),
            )
        except Exception as exc:
            raise TTSGenerationError(
                f"XTTS generation failed: {exc}"
            ) from exc

        if not output_path.exists():
            raise TTSGenerationError(
                f"XTTS did not create the expected output: {output_path}"
            )

        return output_path
