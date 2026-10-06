# Personalized Text-to-Audiobook Converter

This project turns a book and a short reference voice into a chapter-wise audiobook using a modular pipeline.

## Phase status

| Phase | Scope | Status |
|---|---|---|
| 1 | Repo, config, environment checks | Complete |
| 2 | Text extraction and normalization | Complete |
| 3 | Reference voice validation and preprocessing | Complete |
| 4 | Long-form context management | Complete |
| 5 | TTS engine abstraction and F5-TTS integration layer | Implemented with lazy dependency checks |
| 6 | Audio assembly and chapter export | Implemented |
| 7 | End-to-end pipeline orchestration | Implemented |
| 8 | Streamlit UI | Implemented |
| 9 | Evaluation module | Implemented, measurement-based and optional |
| 10 | Documentation | Implemented |

## Project scope and honesty
This project uses real pretrained TTS when the dependency is installed. It does not fabricate model results, MOS scores, or speaker-similarity numbers. Any unavailable metric is reported as unavailable rather than guessed.

## Core features
- TXT / PDF / EPUB book extraction
- Normalization and chapter detection
- Paragraph-aware chunking and context carry-over
- Reference voice validation and trimming
- Lazy F5-TTS backend detection
- Chapter and final audiobook WAV assembly
- Optional MP3 export via FFmpeg
- Streamlit interface with clear dependency messaging

## Quick start on Windows (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts/setup_check.py
```

## Optional TTS installation
```powershell
pip install f5-tts
```
If you want GPU acceleration, install the matching PyTorch build for your CUDA version before or after F5-TTS.

## Run the app
```powershell
streamlit run app/main.py
```

## Run the tests
```powershell
python -m pytest -q -m "not integration"
```

## Important limitations
- Real F5-TTS inference is not tested on this machine unless it is installed and run here.
- The long-form style controls are heuristic, not a learned emotional model.
- MP3 export requires FFmpeg to be installed and on PATH.

## Documentation
See the files under `docs/` for setup notes, architecture, evaluation, limitations, and viva guidance.
