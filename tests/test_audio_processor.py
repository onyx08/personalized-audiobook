import numpy as np, pytest, soundfile as sf
from backend.audio_processor import *
from backend.speaker_processor import *

SR = 16000
def tone(sec, sr=SR, f=220, amp=0.3):
    t = np.arange(int(sec * sr)) / sr
    return (amp * np.sin(2 * np.pi * f * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))).astype(np.float32)

def write(tmp_path, name, x, sr=SR):
    p = tmp_path / name; sf.write(str(p), x, sr); return p

# ---- reference validation
def test_valid_reference(tmp_path):
    r = validate_reference_audio(write(tmp_path, "a.wav", tone(8)))
    assert r.ok and 7.9 < r.duration_sec < 8.1 and r.channels == 1

def test_too_short(tmp_path):
    r = validate_reference_audio(write(tmp_path, "a.wav", tone(2)))
    assert not r.ok and any("short" in e for e in r.errors)

def test_silent_and_clipped(tmp_path):
    assert not validate_reference_audio(write(tmp_path, "s.wav", np.zeros(SR * 8, np.float32))).ok
    c = np.clip(tone(8) * 10, -1, 1)
    r = validate_reference_audio(write(tmp_path, "c.wav", c))
    assert not r.ok and r.clipping_pct > 1

def test_bad_files(tmp_path):
    assert not validate_reference_audio(tmp_path / "missing.wav").ok
    bad = tmp_path / "bad.wav"; bad.write_bytes(b"garbage")
    assert not validate_reference_audio(bad).ok
    assert not validate_reference_audio(write(tmp_path, "x.wav", tone(8)).rename(tmp_path / "x.txt")).ok

def test_preprocess_stereo_resample_trim(tmp_path):
    lead = np.zeros(SR, np.float32)
    mono = np.concatenate([lead, tone(8), lead])
    stereo = np.stack([mono, mono * 0.5], axis=1)
    p = write(tmp_path, "st.wav", stereo)
    prof = preprocess_reference_audio(p, tmp_path / "out", name="me", target_sr=24000)
    y, sr = sf.read(prof.audio_path)
    assert sr == 24000 and y.ndim == 1
    assert 7.5 < prof.duration_sec < 8.6            # silence trimmed
    assert abs(np.abs(y).max() - 0.95) < 0.01        # peak normalised
    assert SpeakerProfile.load(tmp_path / "out" / "me_profile.json").name == "me"

def test_preprocess_rejects_invalid(tmp_path):
    with pytest.raises(ReferenceAudioError):
        preprocess_reference_audio(write(tmp_path, "a.wav", tone(1)), tmp_path)

# ---- chunk inspection
def test_inspect_chunk():
    assert inspect_chunk(tone(1), SR).ok
    assert inspect_chunk(np.array([]), SR).problems == ("empty",)
    assert not inspect_chunk(np.array([0.1, np.nan] * 8000, np.float32), SR).ok
    assert not inspect_chunk(np.zeros(SR, np.float32), SR).ok
    assert "clipped" in inspect_chunk(np.clip(tone(1) * 10, -1, 1), SR).problems

# ---- concatenation with mixed sample rates / channels
def test_concatenate_mixed_sr_and_channels():
    a, b = tone(1, 16000), np.stack([tone(1, 22050)] * 2, axis=1)
    out = concatenate([a, b], [16000, 22050], [500], target_sr=24000)
    assert out.ndim == 1
    assert abs(len(out) / 24000 - 2.5) < 0.05        # 1s + 0.5s pause + 1s

def test_concatenate_mismatch_raises():
    with pytest.raises(AudioProcessingError):
        concatenate([tone(1)], [16000, 16000], [])

def test_concatenate_empty():
    assert len(concatenate([], [], [])) == 0

# ---- loudness / trim / export
def test_normalize_loudness_hits_target_and_no_clip():
    x = tone(3, amp=0.01)
    y = normalize_loudness(x, SR, target=-18.0)
    assert abs(measure_loudness(y, SR) - (-18.0)) < 1.0 and np.abs(y).max() <= 0.97 + 1e-6
    loud = tone(3, amp=0.9)
    assert np.abs(normalize_loudness(loud, SR, target=-3.0)).max() <= 0.97 + 1e-6

def test_trim_edges():
    x = np.concatenate([np.zeros(SR, np.float32), tone(1), np.zeros(SR, np.float32)])
    assert len(trim_edges(x, SR)) < len(x) * 0.6

@pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg missing")
def test_wav_to_mp3(tmp_path):
    w = save_wav(tone(2), SR, tmp_path / "a.wav")
    m = wav_to_mp3(w, tmp_path / "a.mp3")
    assert m.exists() and m.stat().st_size > 1000

@pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg missing")
def test_mp3_reference_decodes_via_ffmpeg(tmp_path):
    w = save_wav(tone(8), SR, tmp_path / "a.wav"); m = wav_to_mp3(w, tmp_path / "a.mp3")
    assert validate_reference_audio(m).ok
