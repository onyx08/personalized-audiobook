# Viva notes

## Key points to explain
- The project uses a real pretrained TTS model when available.
- Long-form continuity is handled by a custom context manager and not by pretending hidden-state injection.
- The emotion/energy/pace heuristics are rule-based and clearly documented.
- The TTS backend is abstracted so the project can use a real model or fail gracefully with a setup message.
- Audio assembly and loudness normalization are handled without storing fake or synthetic speech.

## Good evaluation narrative
Explain the pipeline as a modular signal-processing + speech-generation workflow. Emphasize that the project is a practical local processing system, not a foundation-model training project.
