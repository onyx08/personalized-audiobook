# CLAUDE.md - project guide for AI coding agents

Project: Personalized Text-to-Audiobook Converter (college NNDL project).

## Rules
- Use REAL pretrained models only (F5-TTS primary, StyleTTS2 optional). Never fake TTS output.
- Do not train foundation models. Do not claim the project trained them.
- If a model/dependency is missing, raise a clear setup message - never silently substitute.
- Never fabricate benchmark numbers, MOS scores, or speaker-similarity results.
- Keep ACTUAL IMPLEMENTED vs CONCEPTUAL/FUTURE clearly separated in docs.
- No secrets in the repo. Config via env vars (.env.example).
- Automated tests must NOT download large models; mark those `@pytest.mark.integration`.

## Layout
app/ (Streamlit UI + config) | backend/ (pipeline modules) | tests/ | scripts/ | docs/

## Commands
- Env report: `python scripts/setup_check.py`
- Tests: `pytest -m "not integration"`
- UI: `streamlit run app/ui.py`

## Phase status
See README.md "Project status".
