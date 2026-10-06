# Architecture

## Problem statement
The project converts a long-form text source into a personalized audiobook using a reference voice clip. The primary objective is to preserve reading continuity while generating chapter-wise speech with a consistent voice identity.

## Objectives
- Extract and normalize text from TXT, PDF, and EPUB sources.
- Detect chapter boundaries and create chunk-aware text segments.
- Adapt the narration style using lightweight heuristics for emotion, energy, and pace.
- Use a real neural TTS model when available.
- Assemble chapter-level audio and export WAV/MP3 output.
- Keep the system modular and easy to extend.

## System components
1. Input and configuration layer: `app/config.py`, `.env.example`
2. Text pipeline: `backend/text_processor.py`
3. Speaker pipeline: `backend/speaker_processor.py`
4. Long-form context management: `backend/context_manager.py`
5. TTS backend abstraction: `backend/tts_engine.py`, `backend/f5_tts_engine.py`
6. Assembly and orchestration: `backend/pipeline.py`
7. Streamlit UI: `app/main.py`
8. Evaluation: `backend/evaluator.py`

## Data flow
Book input -> text extraction -> normalization -> chapter detection -> paragraph and sentence segmentation -> chunk planning -> reference audio preprocessing -> TTS inference -> chapter assembly -> final audiobook export.

## What is implemented vs conceptual
Implemented: text extraction, normalization, chapter detection, reference voice validation, chunk planning, lazy TTS integration layer, orchestrated assembly, and a Streamlit UI.

Conceptual / future work: learned emotion conditioning, end-to-end neural context injection, stronger speaker embedding evaluation, and full production-grade ASR metrics.
