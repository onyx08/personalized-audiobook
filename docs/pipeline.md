# Pipeline

## 1. Book input
Book files are loaded from the configured input directory. The system supports TXT, PDF, and EPUB.

## 2. Text extraction
`extract_text()` reads the file and decodes it. PDF pages are merged into a single reading order; EPUB chapters are parsed by document spine order.

## 3. Normalization
The text is cleaned by removing bad Unicode characters, preserving punctuation, converting numbers into words, and joining broken lines.

## 4. Chapter segmentation
`detect_chapters()` searches for headings such as "Chapter 1", "Part II", and prologue-like titles. When no heading is found, the book is treated as a single chapter.

## 5. Chunk planning
`LongFormContextManager` splits long paragraphs into sentence-aware chunks and keeps a rolling contextual window to preserve continuity across paragraph transitions.

## 6. Reference voice preparation
A short voice sample is validated, trimmed, resampled to 24 kHz, and normalized. This audio is used as the conditioning reference for the TTS engine when the real backend is available.

## 7. TTS generation
The actual inference step is isolated behind the `BaseTTSEngine` abstraction. `F5TTSEngine` lazily loads the package when installed and raises a clear setup error otherwise.

## 8. Audio assembly
Generated chunk files are merged in chapter order. Controlled pauses are inserted between segments, and chapter files are then combined into a final audiobook export.

## 9. Output
The pipeline writes WAV files by default and may also export MP3 when FFmpeg is available.
