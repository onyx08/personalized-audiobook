# Limitations

## Current technical constraints
- F5-TTS is a real pretrained model, but it is optional and may be unavailable in this environment.
- Long-form style control is heuristic, not a learned emotion model.
- The project is designed for local generation and not optimized for large-scale cloud inference.
- GPU performance depends on the installed PyTorch build and available VRAM.
- MP3 export requires FFmpeg on PATH.

## Honest project scope
This repository implements a working local pipeline for text preprocessing, reference-voice validation, chunk planning, TTS integration, and audio assembly. It does not claim to have benchmarked or certified model quality beyond what is actually measured.
