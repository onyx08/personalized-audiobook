# Evaluation framework

The evaluation module is designed for metrics that can be computed realistically in a college project environment.

## Metrics that can be measured
- Audio duration
- Chunk count
- Processing time
- Speaker-embedding similarity when a speech encoder is installed
- Loudness / clipping / sanity checks

## Optional advanced metrics
- ASR-based intelligibility (requires `openai-whisper`)
- Human evaluation questionnaire for naturalness, prosody, voice similarity, and coherence

## Human evaluation form
Rate each criterion from 1 (poor) to 5 (excellent):
- Voice similarity
- Naturalness
- Intelligibility
- Prosody
- Consistency across chunks
- Overall listening experience

## Important rule
No synthetic, fabricated, or guessed results are reported. If a dependency is missing, the evaluation step records that the metric was unavailable rather than inventing a value.
