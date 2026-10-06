# Setup

## Recommended environment
- Windows 10/11
- Python 3.11
- VS Code
- FFmpeg on PATH for MP3 export

## Installation
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts/setup_check.py
```

## Optional neural TTS setup
```powershell
pip install f5-tts
```
Install the correct PyTorch build for your GPU if you want CUDA inference. CPU is supported when the package is importable, but generation will be slower.

## Run the UI
```powershell
streamlit run app/main.py
```

## Run the tests
```powershell
python -m pytest -q -m "not integration"
```

## Notes
The project intentionally does not force a model download during import. Model loading is lazy and only happens at generation time.
