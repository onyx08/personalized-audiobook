from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from backend.pipeline import BookPipeline, PipelineResult, assemble_chapter_audio


class StubEngine:
    def __init__(self):
        self.name = "stub"
        self.calls = []

    def is_available(self):
        return True

    def load_model(self):
        return object()

    def generate(self, text, reference_audio, output_path, **kwargs):
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"audio")
        self.calls.append(
            {
                "text": text,
                "reference_audio": reference_audio,
                "output_path": str(output_path),
                "kwargs": kwargs,
            }
        )
        return path


def test_assemble_chapter_audio_exports_wav(tmp_path):
    out = assemble_chapter_audio([b"x"], tmp_path / "chapter.wav")
    assert out.exists()


def test_pipeline_runs_without_real_tts(tmp_path):
    ref = tmp_path / "ref.wav"
    t = np.arange(8 * 16000) / 16000
    x = (0.25 * np.sin(2 * np.pi * 220 * t) * (0.7 + 0.3 * np.sin(2 * np.pi * 3 * t))).astype(np.float32)
    sf.write(ref, x, 16000)

    pipeline = BookPipeline(
        book_path=tmp_path / "book.txt",
        reference_audio=ref,
        output_dir=tmp_path / "out",
        ref_text="Hello there. This is the reference speaker.",
        engine=StubEngine(),
    )

    pipeline.book_path.write_text("Chapter 1\n\nHello there. This is a sample chapter.", encoding="utf-8")
    result = pipeline.run()
    assert isinstance(result, PipelineResult)
    assert result.chapter_count >= 1
    assert result.output_dir.exists()
    assert pipeline.engine.calls
    assert all(
        call["kwargs"]["ref_text"] == "Hello there. This is the reference speaker."
        for call in pipeline.engine.calls
    )
    profile = (tmp_path / "out" / "reference" / "speaker_profile.json").read_text(encoding="utf-8")
    assert '"ref_text": "Hello there. This is the reference speaker."' in profile


def test_pipeline_rejects_missing_reference_transcript(tmp_path):
    with pytest.raises(ValueError, match="Reference transcript is required"):
        BookPipeline(
            book_path=tmp_path / "book.txt",
            reference_audio=tmp_path / "ref.wav",
            output_dir=tmp_path / "out",
            ref_text="  ",
            engine=StubEngine(),
        )
