from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import backend.f5_tts_engine as f5_tts_engine
from backend.tts_engine import (
    BaseTTSEngine,
    F5TTSEngine,
    TTSGenerationError,
    TTSUnavailableError,
    detect_tts_backend,
)


class DummyEngine(BaseTTSEngine):
    name = "dummy"

    def is_available(self):
        return True

    def load_model(self):
        return object()

    def generate(self, text, reference_audio, output_path, **kwargs):
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"fake-audio")
        return out


def test_base_engine_requires_implementation():
    eng = DummyEngine()
    assert eng.name == "dummy"
    assert eng.is_available() is True
    assert eng.generate("hello", "ref.wav", "out.wav").exists()


def test_tts_backend_detection_returns_object_or_none():
    backend = detect_tts_backend()
    assert backend is None or hasattr(backend, "name")


def test_f5_engine_reports_clean_error_when_missing(monkeypatch, tmp_path):
    def missing_api(name):
        raise ModuleNotFoundError(f"No module named '{name}'")

    monkeypatch.setattr(f5_tts_engine.importlib, "import_module", missing_api)
    eng = F5TTSEngine(model_dir=tmp_path / "f5-cache", device="cpu")

    assert eng.is_available() is False
    with pytest.raises(TTSUnavailableError, match="F5-TTS is unavailable"):
        eng.ensure_ready()


def test_f5_engine_calls_current_api_and_uses_written_wav(monkeypatch, tmp_path):
    ref_audio = tmp_path / "reference.wav"
    ref_audio.write_bytes(b"reference fixture")
    output = tmp_path / "generated.wav"
    infer = Mock()
    constructor = Mock()

    class MockF5TTS:
        def __init__(self, **kwargs):
            constructor(**kwargs)
            self.infer = infer

    def infer_side_effect(**kwargs):
        Path(kwargs["file_wave"]).write_bytes(b"mock API output")
        return object(), 24000, object()

    infer.side_effect = infer_side_effect
    monkeypatch.setattr(
        f5_tts_engine.importlib,
        "import_module",
        lambda name: SimpleNamespace(F5TTS=MockF5TTS) if name == "f5_tts.api" else None,
    )

    engine = F5TTSEngine(model_dir=tmp_path / "cache", device="cpu")
    assert engine.is_available() is True

    generated = engine.generate(
        "Generated book sentence.",
        ref_audio,
        output,
        ref_text="Exact words from the reference recording.",
        speed=1.1,
    )

    constructor.assert_called_once_with(device="cpu", hf_cache_dir=str(tmp_path / "cache"))
    infer.assert_called_once_with(
        ref_file=str(ref_audio),
        ref_text="Exact words from the reference recording.",
        gen_text="Generated book sentence.",
        speed=1.1,
        cfg_strength=2.0,
        file_wave=str(output),
    )
    assert generated == output
    assert generated.is_file()


def test_f5_engine_requires_explicit_reference_transcript(tmp_path):
    ref_audio = tmp_path / "reference.wav"
    ref_audio.write_bytes(b"reference fixture")
    engine = F5TTSEngine(model_dir=tmp_path / "cache", device="cpu")

    with pytest.raises(TTSGenerationError, match="Reference transcript is required"):
        engine.generate("Generated sentence.", ref_audio, tmp_path / "generated.wav")
