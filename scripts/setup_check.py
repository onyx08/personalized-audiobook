"""Environment report: prints what is and is NOT available. Never installs anything."""
import importlib, platform, shutil, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def mod(name):
    try:
        m = importlib.import_module(name); return True, getattr(m, "__version__", "")
    except Exception as e:
        return False, type(e).__name__

def main():
    print(f"OS      : {platform.platform()}\nPython  : {sys.version.split()[0]}")
    if sys.version_info[:2] != (3, 11):
        print("  note: project targets Python 3.11; other versions may work but F5-TTS is tested on 3.10-3.12")
    for tool in ("ffmpeg", "git", "espeak-ng", "nvidia-smi"):
        print(f"{tool:<8}: {'found ' + shutil.which(tool) if shutil.which(tool) else 'MISSING'}")
    ok, v = mod("torch")
    if ok:
        import torch
        print(f"torch   : {v}  cuda={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            print(f"GPU     : {p.name}, {p.total_memory/1e9:.1f} GB VRAM")
        else:
            print("GPU     : none -> TTS will run on CPU (very slow)")
    else:
        print("torch   : MISSING")
    for name in ("f5_tts", "resemblyzer", "whisper", "spacy", "fitz", "ebooklib", "librosa", "soundfile", "pydub", "streamlit", "num2words"):
        ok, v = mod(name); print(f"{name:<11}: {'OK ' + str(v) if ok else 'MISSING (' + v + ')'}")
    from app.config import get_settings
    s = get_settings(); print(f"\nOutput dir: {s.output_dir}\nModel cache: {s.model_cache_dir}")

if __name__ == "__main__":
    main()
