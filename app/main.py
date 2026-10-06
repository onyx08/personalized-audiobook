from __future__ import annotations

from pathlib import Path

import streamlit as st

from app.config import get_settings
from backend.pipeline import BookPipeline
from backend.tts_engine import detect_tts_backend


# ---------------------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="Personalized Audiobook",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ---------------------------------------------------------------------
# VISUAL STYLE
# ---------------------------------------------------------------------

def _apply_styles() -> None:
    st.markdown(
        """
        <style>

        /* ============================================================
           GLOBAL
        ============================================================ */

        .stApp {
            background-color: #F7F4EE !important;
            color: #263238 !important;
        }

        .main .block-container {
            max-width: 1120px;
            padding-top: 2.8rem;
            padding-bottom: 3rem;
        }

        /* ============================================================
           TYPOGRAPHY
        ============================================================ */

        .project-kicker {
            color: #B56A45 !important;
            font-size: 0.76rem;
            font-weight: 750;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            margin-bottom: 0.45rem;
        }

        .main-title {
            color: #173F4F !important;
            font-size: 2.65rem;
            line-height: 1.08;
            font-weight: 700;
            letter-spacing: -0.035em;
            margin-bottom: 0.45rem;
        }

        .subtitle {
            color: #5D686C !important;
            font-size: 1.03rem;
            line-height: 1.55;
            max-width: 760px;
            margin-bottom: 1.6rem;
        }

        .section-number {
            color: #B56A45 !important;
            font-size: 0.74rem;
            font-weight: 800;
            letter-spacing: 0.16em;
            margin-bottom: 0.18rem;
        }

        .section-title {
            color: #173F4F !important;
            font-size: 1.35rem;
            font-weight: 680;
            margin-bottom: 0.15rem;
        }

        .section-description {
            color: #687377 !important;
            font-size: 0.9rem;
            margin-bottom: 1rem;
        }

        /* ============================================================
           STATUS
        ============================================================ */

        .status-pill {
            display: inline-block;
            padding: 0.36rem 0.72rem;
            border-radius: 999px;
            background: #E2ECEA !important;
            color: #285C60 !important;
            font-size: 0.78rem;
            font-weight: 650;
            border: 1px solid #C9DBD8;
            margin-bottom: 1.7rem;
        }

        .status-dot {
            color: #2F6F73 !important;
            font-size: 0.85rem;
        }

        /* ============================================================
           WORKFLOW CARDS
        ============================================================ */

        .workflow-card {
            background: #EEE8DC !important;
            border: 1px solid #DDD3C2;
            border-radius: 12px;
            padding: 1rem 1.05rem;
            min-height: 118px;
            margin-bottom: 1.6rem;
        }

        .workflow-card-number {
            color: #B56A45 !important;
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0.13em;
            margin-bottom: 0.45rem;
        }

        .workflow-card-title {
            color: #173F4F !important;
            font-size: 1rem;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }

        .workflow-card-text {
            color: #697275 !important;
            font-size: 0.82rem;
            line-height: 1.45;
        }

        /* ============================================================
           STREAMLIT INPUT LABELS
           This fixes the white labels inside Advanced Settings.
        ============================================================ */

        [data-testid="stWidgetLabel"] {
            color: #173F4F !important;
        }

        [data-testid="stWidgetLabel"] p {
            color: #173F4F !important;
            font-weight: 650 !important;
        }

        [data-testid="stWidgetLabel"] label {
            color: #173F4F !important;
        }

        /* Help / description text */

        [data-testid="stWidgetLabel"] small {
            color: #687377 !important;
        }

        /* ============================================================
           TEXT INPUTS + TEXTAREA
           This fixes the dark transcript box.
        ============================================================ */

        input,
        textarea {
            background-color: #FBFAF7 !important;
            color: #263238 !important;
            -webkit-text-fill-color: #263238 !important;
            caret-color: #173F4F !important;
            border-color: #CFC8BB !important;
        }

        input::placeholder,
        textarea::placeholder {
            color: #899295 !important;
            -webkit-text-fill-color: #899295 !important;
            opacity: 1 !important;
        }

        input:focus,
        textarea:focus {
            background-color: #FFFFFF !important;
            color: #263238 !important;
            -webkit-text-fill-color: #263238 !important;
            border-color: #2F6F73 !important;
            box-shadow: 0 0 0 1px #2F6F73 !important;
        }

        /* BaseWeb input wrappers */

        [data-baseweb="input"] {
            background-color: #FBFAF7 !important;
            border-color: #CFC8BB !important;
        }

        [data-baseweb="input"] > div {
            background-color: #FBFAF7 !important;
        }

        [data-baseweb="input"] input {
            background-color: transparent !important;
            color: #263238 !important;
            -webkit-text-fill-color: #263238 !important;
        }

        [data-baseweb="textarea"] {
            background-color: #FBFAF7 !important;
            border-color: #CFC8BB !important;
        }

        [data-baseweb="textarea"] textarea {
            background-color: transparent !important;
            color: #263238 !important;
            -webkit-text-fill-color: #263238 !important;
        }

        /* ============================================================
           SELECT BOX
           This fixes the dark English dropdown.
        ============================================================ */

        [data-baseweb="select"] > div {
            background-color: #FBFAF7 !important;
            color: #263238 !important;
            border-color: #CFC8BB !important;
        }

        [data-baseweb="select"] input {
            color: #263238 !important;
            -webkit-text-fill-color: #263238 !important;
        }

        [data-baseweb="select"] span {
            color: #263238 !important;
        }

        [data-baseweb="select"] svg {
            fill: #173F4F !important;
        }

        /* Dropdown menu */

        [role="listbox"] {
            background-color: #FBFAF7 !important;
        }

        [role="option"] {
            background-color: #FBFAF7 !important;
            color: #263238 !important;
        }

        [role="option"]:hover {
            background-color: #E2ECEA !important;
            color: #173F4F !important;
        }

        /* ============================================================
           EXPANDER
           This fixes the dark Advanced Settings header.
        ============================================================ */

        [data-testid="stExpander"] {
            border: 1px solid #D8D1C4 !important;
            border-radius: 10px !important;
            background-color: #FBFAF7 !important;
            overflow: hidden;
        }

        [data-testid="stExpander"] details {
            background-color: #FBFAF7 !important;
        }

        [data-testid="stExpander"] summary {
            background-color: #EEE8DC !important;
            color: #173F4F !important;
        }

        [data-testid="stExpander"] summary:hover {
            background-color: #E6DECF !important;
        }

        [data-testid="stExpander"] summary span {
            color: #173F4F !important;
        }

        [data-testid="stExpander"] summary p {
            color: #173F4F !important;
            font-weight: 650 !important;
        }

        [data-testid="stExpander"] [data-testid="stExpanderDetails"] {
            background-color: #FBFAF7 !important;
        }

        /* ============================================================
           SLIDER
           Deep teal instead of Streamlit red.
        ============================================================ */

        [data-testid="stSlider"] {
            color: #173F4F !important;
        }

        [data-testid="stSlider"] [role="slider"] {
            background-color: #2F6F73 !important;
            border-color: #2F6F73 !important;
        }

        [data-testid="stSlider"] [data-baseweb="slider"] {
            color: #2F6F73 !important;
        }

        /* Slider value text */

        [data-testid="stSlider"] [data-testid="stThumbValue"] {
            color: #2F6F73 !important;
        }

        /* ============================================================
           CHECKBOX
        ============================================================ */

        [data-testid="stCheckbox"] label {
            color: #263238 !important;
        }

        [data-testid="stCheckbox"] label p {
            color: #263238 !important;
        }

        /* ============================================================
           FILE UPLOADER
        ============================================================ */

        [data-testid="stFileUploader"] {
            background-color: #FBFAF7 !important;
            border-radius: 10px !important;
        }

        [data-testid="stFileUploader"] section {
            background-color: #FBFAF7 !important;
            border-color: #CFC8BB !important;
        }

        [data-testid="stFileUploader"] section > div {
            background-color: #FBFAF7 !important;
        }

        [data-testid="stFileUploader"] p {
            color: #5D686C !important;
        }

        [data-testid="stFileUploader"] small {
            color: #7A8588 !important;
        }

        [data-testid="stFileUploader"] button {
            color: #173F4F !important;
            border-color: #AEBDBA !important;
            background-color: #E2ECEA !important;
        }

        /* ============================================================
           BUTTON
        ============================================================ */

        div.stButton > button {
            background-color: #2F6F73 !important;
            color: #FFFFFF !important;
            border: 1px solid #2F6F73 !important;
            border-radius: 9px !important;
            min-height: 2.8rem;
            font-weight: 700;
            letter-spacing: 0.01em;
        }

        div.stButton > button:hover {
            background-color: #245C60 !important;
            border-color: #245C60 !important;
            color: #FFFFFF !important;
        }

        div.stButton > button p {
            color: #FFFFFF !important;
        }

        /* ============================================================
           CAPTIONS / HELP TEXT
        ============================================================ */

        .stCaption,
        [data-testid="stCaptionContainer"] {
            color: #687377 !important;
        }

        /* ============================================================
           ALERTS
        ============================================================ */

        [data-testid="stAlert"] {
            border-radius: 9px !important;
        }

        /* ============================================================
           RESULT PANEL
        ============================================================ */

        .result-panel {
            background: #E6EFED !important;
            border: 1px solid #C9DBD8;
            border-radius: 14px;
            padding: 1.35rem;
            margin-top: 1.25rem;
            margin-bottom: 1.25rem;
        }

        .result-kicker {
            color: #2F6F73 !important;
            font-size: 0.74rem;
            font-weight: 800;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
        }

        .result-title {
            color: #173F4F !important;
            font-size: 1.55rem;
            font-weight: 720;
            margin-bottom: 0.25rem;
        }

        .result-subtitle {
            color: #5F6B6E !important;
            font-size: 0.9rem;
            margin-bottom: 1.2rem;
        }

        .metric-card {
            background: #FBFAF7 !important;
            border: 1px solid #D6DFDD;
            border-radius: 10px;
            padding: 0.8rem 0.9rem;
        }

        .metric-label {
            color: #788184 !important;
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        .metric-value {
            color: #173F4F !important;
            font-size: 1.08rem;
            font-weight: 700;
            margin-top: 0.18rem;
        }

        /* ============================================================
           INFORMATION / FOOTER
        ============================================================ */

        .info-heading {
            color: #173F4F !important;
            font-size: 1.15rem;
            font-weight: 680;
            margin-bottom: 0.55rem;
        }

        .info-text {
            color: #606C70 !important;
            font-size: 0.88rem;
            line-height: 1.65;
        }

        .pipeline-line {
            color: #173F4F !important;
            font-size: 0.9rem;
            font-weight: 600;
            line-height: 1.7;
        }

        .pipeline-arrow {
            color: #B56A45 !important;
            padding: 0 0.25rem;
        }

        .project-footer {
            color: #7A8284 !important;
            font-size: 0.76rem;
            text-align: center;
            line-height: 1.6;
            padding-top: 1.6rem;
        }

        /* ============================================================
           REDUCE DEFAULT SPACING
        ============================================================ */

        .element-container {
            margin-bottom: 0.25rem;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def _audio_type(path: Path) -> str:
    if path.suffix.lower() == ".mp3":
        return "audio/mp3"
    return "audio/wav"


def _render_setup_message() -> None:
    st.error("No usable TTS backend was detected.")

    st.markdown(
        """
        This project uses **XTTS-v2** for reference-voice conditioned
        narration.

        Launch Streamlit from the XTTS environment:

        ```powershell
        .\\.venv-xtts\\Scripts\\python.exe -m streamlit run .\\app\\ui.py
        ```
        """
    )


def _render_header(backend) -> None:
    st.markdown(
        '<div class="project-kicker">NNDL Project · Speech Synthesis</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="main-title">Personalized Audiobook</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">'
        "Reference-voice conditioned text-to-speech for long-form narration. "
        "Upload a book, provide a short voice sample, and generate a "
        "personalized audiobook."
        "</div>",
        unsafe_allow_html=True,
    )

    if backend is not None:
        st.markdown(
            '<div class="status-pill">'
            '<span class="status-dot">●</span>&nbsp; XTTS-v2 ready'
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="status-pill">'
            '<span class="status-dot">●</span>&nbsp; TTS backend unavailable'
            "</div>",
            unsafe_allow_html=True,
        )


def _render_workflow() -> None:
    cols = st.columns(3)

    workflow = [
        (
            "01",
            "BOOK",
            "Upload the source book in TXT, PDF, or EPUB format.",
        ),
        (
            "02",
            "VOICE",
            "Provide a clear short reference recording and its transcript.",
        ),
        (
            "03",
            "NARRATE",
            "Generate chapter-wise narration using the reference voice.",
        ),
    ]

    for col, (number, title, description) in zip(cols, workflow):
        with col:
            st.markdown(
                f"""
                <div class="workflow-card">
                    <div class="workflow-card-number">{number}</div>
                    <div class="workflow-card-title">{title}</div>
                    <div class="workflow-card-text">{description}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _render_how_it_works() -> None:
    st.markdown("---")

    st.markdown(
        '<div class="info-heading">How the system works</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="pipeline-line">
            Book
            <span class="pipeline-arrow">→</span>
            Text processing
            <span class="pipeline-arrow">→</span>
            Chapter &amp; context management
            <span class="pipeline-arrow">→</span>
            Reference-voice TTS
            <span class="pipeline-arrow">→</span>
            Audio assembly
            <span class="pipeline-arrow">→</span>
            Audiobook
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="info-text">
        The system extracts and normalizes book text, identifies chapters
        and narration segments, preprocesses a short reference voice sample,
        generates speech using XTTS-v2, and assembles the generated segments
        into chapter-level and complete audiobook audio.
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_footer() -> None:
    st.markdown(
        """
        <div class="project-footer">
            <b>Personalized Audiobook</b><br>
            NNDL Project · Reference-Voice Conditioned Speech Synthesis
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------
# MAIN APPLICATION
# ---------------------------------------------------------------------

def main() -> None:
    settings = get_settings()

    _apply_styles()

    backend = detect_tts_backend()

    _render_header(backend)

    # ================================================================
    # WORKFLOW OVERVIEW
    # ================================================================

    _render_workflow()

    # ================================================================
    # 01 — BOOK
    # ================================================================

    st.markdown(
        '<div class="section-number">01 · BOOK</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-title">Choose your book</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-description">'
        "Upload the source material you want to convert into an audiobook."
        "</div>",
        unsafe_allow_html=True,
    )

    book = st.file_uploader(
        "Upload book",
        type=["txt", "pdf", "epub"],
        label_visibility="collapsed",
        help="Supported formats: TXT, PDF and EPUB.",
    )

    if book is not None:
        st.success(f"Book loaded · {book.name}")

    # ================================================================
    # 02 — VOICE
    # ================================================================

    st.markdown(
        '<div class="section-number">02 · VOICE</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-title">Provide the reference voice</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-description">'
        "Use a clear 6–10 second recording of the speaker whose voice "
        "should be used for narration."
        "</div>",
        unsafe_allow_html=True,
    )

    voice_col, transcript_col = st.columns(2)

    with voice_col:
        st.markdown(
            '<div class="input-label">Reference audio</div>',
            unsafe_allow_html=True,
        )

        ref_audio = st.file_uploader(
            "Reference audio",
            type=["wav", "mp3", "flac", "ogg", "m4a"],
            label_visibility="collapsed",
            help=(
                "A clear recording with minimal background noise works best."
            ),
        )

        if ref_audio is not None:
            st.success(f"Reference sample loaded · {ref_audio.name}")

    with transcript_col:
        st.markdown(
            '<div class="input-label">Reference transcript</div>',
            unsafe_allow_html=True,
        )

        ref_text = st.text_area(
            "Reference transcript",
            placeholder=(
                "Type exactly what is spoken in the reference recording..."
            ),
            label_visibility="collapsed",
            height=120,
            help=(
                "Enter the exact words spoken in the reference audio."
            ),
        )

    # ================================================================
    # ADVANCED SETTINGS
    # ================================================================

    with st.expander("⚙  Advanced Settings"):
        st.caption(
            "Technical controls are hidden from the normal workflow but "
            "can be expanded during development, testing, or demonstration."
        )

        output_dir = st.text_input(
            "Output directory",
            value=str(settings.output_dir),
            help="Folder where generated audiobook files are saved.",
        )

        max_chunk_chars = st.slider(
            "Narration segment size",
            min_value=200,
            max_value=600,
            value=settings.max_chunk_chars,
            step=20,
            help=(
                "Maximum approximate character length for internal "
                "narration segments."
            ),
        )

        language = st.selectbox(
            "Narration language",
            options=["en"],
            index=0,
            format_func=lambda value: "English",
        )

        show_debug = st.checkbox(
            "Show technical generation details",
            value=False,
        )

        st.markdown("---")

        if backend is not None:
            st.caption(
                f"Backend: {backend.describe()}"
            )
        else:
            st.caption("Backend: unavailable")

        st.caption(
            f"Default output: {settings.output_dir}"
        )

    # ================================================================
    # 03 — NARRATION
    # ================================================================

    st.markdown(
        '<div class="section-number">03 · NARRATE</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-title">Generate your audiobook</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-description">'
        "The uploaded book will be segmented, synthesized with the "
        "reference voice, and assembled into a complete audiobook."
        "</div>",
        unsafe_allow_html=True,
    )

    generate_btn = st.button(
        "Generate Personalized Audiobook  →",
        type="primary",
        use_container_width=True,
    )

    # ================================================================
    # GENERATION
    # ================================================================

    if generate_btn:

        if book is None:
            st.error(
                "Please upload a book before generating the audiobook."
            )
            return

        if ref_audio is None:
            st.error(
                "Please upload a reference voice sample."
            )
            return

        if not ref_text.strip():
            st.error(
                "Please enter the exact transcript of the reference "
                "voice sample."
            )
            return

        if backend is None:
            _render_setup_message()
            return

        # ------------------------------------------------------------
        # Save uploaded files
        # ------------------------------------------------------------

        settings.input_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        book_path = settings.input_dir / book.name
        book_path.write_bytes(book.getvalue())

        ref_path = settings.reference_dir / ref_audio.name
        ref_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        ref_path.write_bytes(ref_audio.getvalue())

        out_dir = Path(output_dir)

        # ------------------------------------------------------------
        # Create pipeline
        # ------------------------------------------------------------

        pipeline = BookPipeline(
            book_path=book_path,
            reference_audio=ref_path,
            output_dir=out_dir,
            ref_text=ref_text,
            engine=backend,
            max_chunk_chars=max_chunk_chars,
        )

        st.markdown("---")

        # ------------------------------------------------------------
        # Generation status
        # ------------------------------------------------------------

        with st.status(
            "Generating your personalized audiobook...",
            expanded=True,
        ) as status:

            st.write("✓ Book loaded")
            st.write("✓ Reference voice loaded")
            st.write("✓ Preparing narration segments")
            st.write("◉ Generating voice-conditioned narration")
            st.write("○ Assembling chapter audio")
            st.write("○ Creating complete audiobook")

            try:
                result = pipeline.run()

            except Exception as exc:
                status.update(
                    label="Audiobook generation failed",
                    state="error",
                )

                st.error(
                    f"Generation failed: {exc}"
                )
                return

            status.update(
                label="Audiobook generation complete",
                state="complete",
            )

        # ============================================================
        # RESULT
        # ============================================================

        st.markdown(
            """
            <div class="result-panel">
                <div class="result-kicker">Generation complete</div>
                <div class="result-title">Audiobook ready</div>
                <div class="result-subtitle">
                    Your reference-voice conditioned narration has been
                    successfully generated.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        metric_1, metric_2, metric_3 = st.columns(3)

        with metric_1:
            st.markdown(
                """
                <div class="metric-card">
                    <div class="metric-label">Chapters</div>
                    <div class="metric-value">
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                f"{result.chapter_count}",
                unsafe_allow_html=True,
            )
            st.markdown("</div></div>", unsafe_allow_html=True)

        with metric_2:
            st.markdown(
                """
                <div class="metric-card">
                    <div class="metric-label">Narration segments</div>
                    <div class="metric-value">
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                f"{result.chunk_count}",
                unsafe_allow_html=True,
            )
            st.markdown("</div></div>", unsafe_allow_html=True)

        with metric_3:
            st.markdown(
                """
                <div class="metric-card">
                    <div class="metric-label">Voice</div>
                    <div class="metric-value">
                        Reference voice
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ------------------------------------------------------------
        # Complete audiobook
        # ------------------------------------------------------------

        final_path = out_dir / "audiobook_complete.wav"

        if final_path.exists():

            st.markdown("### Personalized Audiobook")

            st.audio(
                str(final_path),
                format=_audio_type(final_path),
            )

            st.download_button(
                label="Download Audiobook",
                data=final_path.read_bytes(),
                file_name="personalized_audiobook.wav",
                mime=_audio_type(final_path),
                use_container_width=True,
            )

        else:
            st.warning(
                "Generation completed, but the final audiobook file "
                "could not be found."
            )

        # ------------------------------------------------------------
        # Chapter-level audio
        # ------------------------------------------------------------

        chapter_files = sorted(
            out_dir.glob("chapter_*.wav")
        )

        if chapter_files:

            st.markdown("### Chapters")

            for chapter_path in chapter_files:

                with st.expander(
                    chapter_path.stem
                ):
                    st.audio(
                        str(chapter_path),
                        format=_audio_type(chapter_path),
                    )

        # ------------------------------------------------------------
        # Debug information
        # ------------------------------------------------------------

        if show_debug:

            st.markdown("---")

            with st.expander(
                "Technical generation details",
                expanded=True,
            ):

                st.write(
                    "Status:",
                    result.status,
                )

                st.write(
                    "Output directory:",
                    str(result.output_dir),
                )

                st.write(
                    "Chapter count:",
                    result.chapter_count,
                )

                st.write(
                    "Narration segment count:",
                    result.chunk_count,
                )

                st.write(
                    "Backend:",
                    backend.describe(),
                )

                st.write(
                    "Language:",
                    language,
                )

                st.write(
                    "Segment size:",
                    max_chunk_chars,
                )

                if result.warnings:
                    st.write("Warnings:")

                    for warning in result.warnings:
                        st.write(
                            f"- {warning}"
                        )

                st.write("Generated files:")

                for generated_file in result.generated_files:
                    st.code(
                        generated_file
                    )

    # ================================================================
    # HOW IT WORKS
    # ================================================================

    _render_how_it_works()

    _render_footer()


# ---------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------

if __name__ == "__main__":
    main()

