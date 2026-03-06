"""
Streamlit UI — sidebar, input modes, HITL forms, results display.
"""

from io import BytesIO

import re
import streamlit as st

from input_handlers import extract_text_from_image, extract_text_from_audio
from pipeline import reset_pipeline, run_pipeline
from agents import log_agent
from memory import memory_save


# ────────────────────────────────────────────────────────────────
# Main page
# ────────────────────────────────────────────────────────────────

def apply_aiplanet_theme():
    """Apply a clean, professional webapp theme across the Streamlit app."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

        :root {
            --ap-bg: #f0f4f8;
            --ap-surface: #ffffff;
            --ap-surface-2: #f8fafc;
            --ap-primary: #2563eb;
            --ap-primary-hover: #1d4ed8;
            --ap-primary-light: #eff6ff;
            --ap-primary-border: #bfdbfe;
            --ap-success: #16a34a;
            --ap-success-bg: #f0fdf4;
            --ap-warning: #d97706;
            --ap-warning-bg: #fffbeb;
            --ap-danger: #dc2626;
            --ap-danger-bg: #fef2f2;
            --ap-text: #0f172a;
            --ap-text-2: #334155;
            --ap-muted: #64748b;
            --ap-muted-2: #94a3b8;
            --ap-border: #e2e8f0;
            --ap-border-strong: #cbd5e1;
            --ap-shadow-sm: 0 1px 3px rgba(15,23,42,0.08), 0 1px 2px rgba(15,23,42,0.06);
            --ap-shadow: 0 4px 16px rgba(15,23,42,0.08), 0 2px 6px rgba(15,23,42,0.06);
            --ap-shadow-lg: 0 10px 36px rgba(15,23,42,0.12), 0 4px 12px rgba(15,23,42,0.08);
            --ap-radius: 12px;
            --ap-radius-sm: 8px;
        }

        * { box-sizing: border-box; }

        .stApp {
            color: var(--ap-text);
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: var(--ap-bg);
        }

        .block-container {
            max-width: 1100px !important;
            padding-top: 0 !important;
            padding-bottom: 2rem !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
        }

        [data-testid="stAppViewContainer"] {
            background: linear-gradient(160deg, #f0f4f8 0%, #e8eef7 50%, #edf2fb 100%);
            min-height: 100vh;
        }

        [data-testid="stHeader"] { background: transparent !important; }

        [data-testid="stSidebar"] {
            background: var(--ap-surface) !important;
            border-right: 1px solid var(--ap-border);
            box-shadow: 2px 0 12px rgba(15,23,42,0.05);
        }

        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--ap-muted);
            font-weight: 700;
            margin-bottom: 0.5rem;
        }

        h1, h2, h3, h4, h5, h6 {
            font-family: 'Inter', sans-serif;
            color: var(--ap-text);
            letter-spacing: -0.025em;
            font-weight: 700;
        }

        /* ── HERO ── */
        .ap-hero {
            background: linear-gradient(135deg, #1e40af 0%, #2563eb 45%, #3b82f6 100%);
            border-radius: 20px;
            padding: 2.2rem 2.4rem 2rem;
            margin: 0.5rem 0 1.4rem 0;
            position: relative;
            overflow: hidden;
        }
        .ap-hero::before {
            content: '';
            position: absolute;
            top: -60px; right: -60px;
            width: 280px; height: 280px;
            background: radial-gradient(circle, rgba(255,255,255,0.12) 0%, transparent 70%);
            border-radius: 50%;
        }
        .ap-hero::after {
            content: '';
            position: absolute;
            bottom: -80px; left: 30%;
            width: 320px; height: 320px;
            background: radial-gradient(circle, rgba(255,255,255,0.07) 0%, transparent 60%);
            border-radius: 50%;
        }

        .ap-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.3rem;
            font-size: 0.72rem;
            font-weight: 700;
            color: rgba(255,255,255,0.95);
            background: rgba(255,255,255,0.18);
            backdrop-filter: blur(8px);
            border: 1px solid rgba(255,255,255,0.3);
            border-radius: 999px;
            padding: 0.22rem 0.7rem;
            margin-bottom: 0.9rem;
            letter-spacing: 0.07em;
            text-transform: uppercase;
        }

        .ap-hero h1 {
            color: #ffffff !important;
            font-size: 2.1rem;
            font-weight: 800;
            letter-spacing: -0.03em;
            margin: 0 0 0.55rem 0;
            line-height: 1.15;
        }

        .ap-subtitle {
            color: rgba(255,255,255,0.82);
            font-size: 1rem;
            line-height: 1.6;
            max-width: 600px;
            margin: 0;
        }

        /* ── STATUS BAR ── */
        .ap-statusbar {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.75rem;
            margin: 0 0 1.5rem 0;
        }

        .ap-pill {
            background: var(--ap-surface);
            border: 1px solid var(--ap-border);
            border-radius: var(--ap-radius);
            padding: 0.75rem 1rem;
            box-shadow: var(--ap-shadow-sm);
            display: flex;
            flex-direction: column;
            gap: 0.2rem;
            transition: box-shadow 0.2s;
        }

        .ap-pill:hover { box-shadow: var(--ap-shadow); }

        .ap-pill-label {
            font-size: 0.7rem;
            font-weight: 700;
            color: var(--ap-muted-2);
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        .ap-pill-value {
            font-size: 0.95rem;
            font-weight: 700;
            color: var(--ap-text);
            line-height: 1.2;
        }

        /* ── SECTION CARDS ── */
        .ap-card {
            background: var(--ap-surface);
            border: 1px solid var(--ap-border);
            border-radius: var(--ap-radius);
            padding: 1.5rem 1.6rem;
            margin-bottom: 1rem;
            box-shadow: var(--ap-shadow-sm);
        }

        .ap-section-heading {
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--ap-muted);
            margin-bottom: 0.85rem;
            display: flex;
            align-items: center;
            gap: 0.4rem;
        }

        /* ── BUTTONS ── */
        .stButton > button {
            font-family: 'Inter', sans-serif;
            font-weight: 600;
            font-size: 0.88rem;
            border-radius: var(--ap-radius-sm);
            border: 1.5px solid var(--ap-border-strong);
            background: var(--ap-surface);
            color: var(--ap-text-2);
            padding: 0.45rem 1.1rem;
            transition: all 0.15s ease;
            letter-spacing: -0.01em;
        }

        .stButton > button:hover {
            border-color: var(--ap-primary);
            color: var(--ap-primary);
            background: var(--ap-primary-light);
            box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
            transform: translateY(-1px);
        }

        /* Primary (Solve) button */
        .stButton > button[kind="primary"],
        .stButton > button[data-testid*="primary"] {
            background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
            color: #ffffff !important;
            border: none;
            box-shadow: 0 2px 8px rgba(37, 99, 235, 0.35);
            font-weight: 700;
            padding: 0.52rem 1.4rem;
        }

        .stButton > button[kind="primary"]:hover {
            background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%);
            box-shadow: 0 4px 16px rgba(37, 99, 235, 0.45);
            transform: translateY(-2px);
            color: #ffffff !important;
        }

        /* ── INPUTS ── */
        .stTextInput > div > div > input,
        .stTextArea textarea {
            font-family: 'Inter', sans-serif;
            font-size: 0.9rem;
            color: var(--ap-text);
            background: var(--ap-surface);
            border: 1.5px solid var(--ap-border);
            border-radius: var(--ap-radius-sm);
            transition: border-color 0.15s, box-shadow 0.15s;
        }

        .stTextInput > div > div > input:focus,
        .stTextArea textarea:focus {
            border-color: var(--ap-primary) !important;
            box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.12) !important;
        }

        [data-baseweb="input"] > div {
            background: var(--ap-surface);
            border: 1.5px solid var(--ap-border);
            border-radius: var(--ap-radius-sm);
        }

        /* ── FILE UPLOADER ── */
        [data-testid="stFileUploader"] section {
            background: var(--ap-primary-light);
            border: 2px dashed var(--ap-primary-border);
            border-radius: var(--ap-radius);
            transition: background 0.15s, border-color 0.15s;
        }

        [data-testid="stFileUploader"] section:hover {
            background: #dbeafe;
            border-color: var(--ap-primary);
        }

        /* Drag-and-drop inner text — must be visible, no strikethrough */
        [data-testid="stFileUploader"] section span,
        [data-testid="stFileUploader"] section small,
        [data-testid="stFileUploader"] section p,
        [data-testid="stFileUploader"] section div {
            color: var(--ap-text-2) !important;
            text-decoration: none !important;
            font-size: 0.88rem !important;
        }

        /* Browse files button */
        [data-testid="stFileUploader"] section button {
            background: var(--ap-primary) !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: var(--ap-radius-sm) !important;
            font-weight: 600 !important;
            font-size: 0.85rem !important;
        }

        /* ── FORM LABELS — outer labels only (not inner uploader content) ── */
        [data-testid="stRadio"] > label,
        [data-testid="stCheckbox"] > label,
        [data-testid="stSelectbox"] > label,
        [data-testid="stTextInput"] > label,
        [data-testid="stTextArea"] > label,
        [data-testid="stNumberInput"] > label,
        [data-testid="stFileUploader"] > label {
            color: var(--ap-text) !important;
            font-size: 0.88rem !important;
            font-weight: 500 !important;
        }

        /* ── RADIO BUTTONS ── */
        [data-testid="stRadio"] > div { gap: 0.4rem; }

        [data-testid="stRadio"] [data-baseweb="radio"] label,
        [data-testid="stRadio"] [data-baseweb="radio"] label p,
        [data-testid="stRadio"] [data-baseweb="radio"] label span {
            color: var(--ap-text) !important;
            font-weight: 500 !important;
            font-size: 0.9rem !important;
            text-decoration: none !important;
        }

        /* Radio circle outline */
        [data-testid="stRadio"] [data-baseweb="radio"] > div:first-child {
            border-color: var(--ap-border-strong) !important;
        }

        /* ── TABS ── */
        [data-testid="stTabs"] [data-baseweb="tab-list"] {
            background: var(--ap-surface-2);
            border-radius: var(--ap-radius-sm) var(--ap-radius-sm) 0 0;
            border-bottom: 2px solid var(--ap-border);
            gap: 0;
        }

        [data-testid="stTabs"] button {
            font-family: 'Inter', sans-serif;
            font-size: 0.88rem;
            font-weight: 600;
            color: var(--ap-muted);
            background: transparent;
            border-radius: 0;
            padding: 0.65rem 1.1rem;
            transition: color 0.15s;
        }

        [data-testid="stTabs"] button:hover { color: var(--ap-primary); }

        [data-testid="stTabs"] button[aria-selected="true"] {
            color: var(--ap-primary);
            font-weight: 700;
        }

        [data-testid="stTabs"] [data-baseweb="tab-highlight"] {
            background: var(--ap-primary);
            height: 2.5px;
            border-radius: 2px;
        }

        /* ── ALERTS ── */
        [data-testid="stAlert"] {
            border-radius: var(--ap-radius-sm);
            border-width: 1px;
            font-size: 0.9rem;
        }

        /* success override */
        div[data-testid="stAlert"][data-baseweb-class*="success"],
        .stAlert .st-emotion-cache-j7aa1c {
            background: var(--ap-success-bg);
            border-color: #86efac;
        }

        /* ── SPINNER ── */
        [data-testid="stSpinner"] span { color: var(--ap-primary) !important; }

        /* ── EXPANDERS ── */
        [data-testid="stExpander"] summary {
            font-weight: 600;
            font-size: 0.9rem;
            color: var(--ap-text-2);
        }

        /* ── DIVIDERS ── */
        hr { border-color: var(--ap-border) !important; margin: 1.2rem 0 !important; }

        /* ── CAPTIONS ── */
        [data-testid="stCaptionContainer"],
        [data-testid="stCaptionContainer"] p { color: var(--ap-muted) !important; font-size: 0.8rem; }

        /* ── MARKDOWN TEXT — keep body text dark, not muted ── */
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li,
        [data-testid="stMarkdownContainer"] span {
            color: var(--ap-text-2);
        }

        /* Help text (tooltip trigger) */
        [data-testid="stTooltipIcon"] { color: var(--ap-muted) !important; }

        /* ── SUBHEADER OVERRIDE ── */
        [data-testid="stMarkdownContainer"] h2 {
            font-size: 1.15rem;
            border-bottom: 1px solid var(--ap-border);
            padding-bottom: 0.4rem;
            margin-bottom: 0.8rem;
        }

        /* ── WARNING BOXES ── */
        div[data-testid="stAlert"] > div {
            padding: 0.75rem 1rem;
        }

        @media (max-width: 768px) {
            .block-container {
                padding-left: 0.75rem !important;
                padding-right: 0.75rem !important;
            }
            .ap-hero { padding: 1.4rem 1.2rem 1.2rem; }
            .ap-hero h1 { font-size: 1.55rem; }
            .ap-statusbar { grid-template-columns: 1fr; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_page_hero():
    """Render a visually strong top hero section."""
    st.markdown(
        """
        <section class="ap-hero">
            <span class="ap-badge">&#x2728; AI Math Mentor</span>
            <h1>Multimodal Math Mentor</h1>
            <p class="ap-subtitle">
                Solve JEE-level problems from text, image, or audio using a
                5-agent pipeline with RAG retrieval and human-in-the-loop checks.
            </p>
            <div style="display:flex;gap:1.4rem;margin-top:1.4rem;flex-wrap:wrap">
                <div style="color:rgba(255,255,255,0.9);font-size:0.82rem;font-weight:600;display:flex;align-items:center;gap:0.35rem">
                    <span style="width:7px;height:7px;background:#4ade80;border-radius:50%;display:inline-block"></span>
                    GPT-4o-mini powered
                </div>
                <div style="color:rgba(255,255,255,0.9);font-size:0.82rem;font-weight:600;display:flex;align-items:center;gap:0.35rem">
                    <span style="width:7px;height:7px;background:#60a5fa;border-radius:50%;display:inline-block"></span>
                    RAG + Memory
                </div>
                <div style="color:rgba(255,255,255,0.9);font-size:0.82rem;font-weight:600;display:flex;align-items:center;gap:0.35rem">
                    <span style="width:7px;height:7px;background:#f9a8d4;border-radius:50%;display:inline-block"></span>
                    HITL Verification
                </div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_top_status_bar():
    """Render compact color-coded status pills."""
    mode = st.session_state.get("input_mode") or "Text"
    stage = st.session_state.get("hitl_stage")
    complete = st.session_state.get("pipeline_complete", False)
    if stage:
        pipeline_state = "Awaiting Input"
        state_color = "#d97706"
        state_bg = "#fffbeb"
        state_dot = "#f59e0b"
    elif complete:
        pipeline_state = "Complete"
        state_color = "#16a34a"
        state_bg = "#f0fdf4"
        state_dot = "#4ade80"
    else:
        pipeline_state = "Ready"
        state_color = "#2563eb"
        state_bg = "#eff6ff"
        state_dot = "#60a5fa"

    raw_conf = st.session_state.get("extraction_confidence")
    conf_display = (raw_conf or "–").capitalize()
    conf_color = {"high": "#16a34a", "medium": "#d97706", "low": "#dc2626"}.get(raw_conf, "#64748b")

    mode_icons = {"Text": "T", "Image": "I", "Audio": "A"}
    mode_icon = mode_icons.get(mode, "T")

    st.markdown(
        f"""
        <div class="ap-statusbar">
            <div class="ap-pill">
                <div class="ap-pill-label">Input Mode</div>
                <div class="ap-pill-value" style="display:flex;align-items:center;gap:0.45rem">
                    <span style="width:24px;height:24px;border-radius:6px;background:#eff6ff;color:#2563eb;font-size:0.7rem;font-weight:800;display:inline-flex;align-items:center;justify-content:center">{mode_icon}</span>
                    {mode}
                </div>
            </div>
            <div class="ap-pill" style="background:{state_bg};border-color:{state_dot}40">
                <div class="ap-pill-label">Pipeline State</div>
                <div class="ap-pill-value" style="color:{state_color};display:flex;align-items:center;gap:0.4rem">
                    <span style="width:8px;height:8px;border-radius:50%;background:{state_dot};display:inline-block"></span>
                    {pipeline_state}
                </div>
            </div>
            <div class="ap-pill">
                <div class="ap-pill-label">Extraction Confidence</div>
                <div class="ap-pill-value" style="color:{conf_color}">{conf_display}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_sidebar(default_api_key: str | None, free_left: int = 0, trial_active: bool = False) -> str:
    """Render sidebar (free trial counter, API key input, history) and return the user-supplied key."""
    with st.sidebar:
        st.markdown(
            '<p style="font-size:1.1rem;font-weight:800;color:#0f172a;letter-spacing:-0.02em;margin-bottom:0.1rem">Math Mentor</p>'
            '<p style="font-size:0.78rem;color:#64748b;margin-bottom:1.2rem">Powered by GPT-4o-mini</p>',
            unsafe_allow_html=True,
        )

        # Free trial banner
        if trial_active:
            pct = int((free_left / 10) * 100)
            bar_color = "#16a34a" if free_left > 5 else ("#d97706" if free_left > 2 else "#dc2626")
            st.markdown(
                f'<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:0.75rem 0.9rem;margin-bottom:1rem">'
                f'<p style="font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.07em;color:#64748b;margin:0 0 0.4rem 0">Free Trial</p>'
                f'<p style="font-size:1.4rem;font-weight:800;color:{bar_color};margin:0 0 0.35rem 0">{free_left}<span style="font-size:0.8rem;font-weight:500;color:#94a3b8"> / 10 solves left</span></p>'
                f'<div style="background:#e2e8f0;border-radius:999px;height:5px">'
                f'<div style="background:{bar_color};width:{pct}%;height:5px;border-radius:999px;transition:width 0.4s"></div>'
                f'</div></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="background:#fef2f2;border:1px solid #fca5a5;border-radius:10px;padding:0.65rem 0.9rem;margin-bottom:1rem">'
                '<p style="font-size:0.82rem;font-weight:600;color:#dc2626;margin:0">Free trial used up.<br>'
                '<span style="font-weight:400;color:#6b7280">Enter your own API key below.</span></p>'
                '</div>',
                unsafe_allow_html=True,
            )

        st.markdown('<p style="font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.07em;color:#94a3b8;margin-bottom:0.3rem">Your API Key</p>', unsafe_allow_html=True)
        api_key = st.text_input(
            "OpenAI API Key",
            value="",
            type="password",
            help="Paste your OpenAI API key. It is NOT stored anywhere.",
            placeholder="sk-...  (optional during free trial)",
            label_visibility="collapsed",
        )
        if api_key:
            st.session_state["using_own_key"] = True
        st.divider()
        st.markdown('<p style="font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.07em;color:#94a3b8;margin-bottom:0.3rem">Solved History</p>', unsafe_allow_html=True)
        if st.session_state.history:
            for i, item in enumerate(reversed(st.session_state.history), 1):
                with st.expander(f"Q{i}: {item['question'][:55]}…"):
                    st.markdown(item["answer"])
        else:
            st.markdown(
                '<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:0.75rem 0.9rem;font-size:0.83rem;color:#94a3b8;text-align:center">No solved problems yet.</div>',
                unsafe_allow_html=True,
            )
    return api_key


def render_input() -> tuple[str, str | None]:
    """Render the input-mode selector and return (raw_text, confidence)."""
    st.markdown(
        '<p style="font-size:0.72rem;font-weight:700;text-transform:uppercase;'
        'letter-spacing:0.08em;color:#64748b;margin:0 0 0.5rem 0">Problem Input</p>',
        unsafe_allow_html=True,
    )
    mode = st.radio(
        "Choose input mode",
        ["Text", "Image", "Audio"],
        horizontal=True,
        label_visibility="collapsed",
    )

    raw_text = ""
    extraction_conf = None

    # ── TEXT ──
    if mode == "Text":
        raw_text = st.text_area(
            "Problem",
            height=130,
            placeholder="e.g.  Solve for x:  2x² + 3x − 5 = 0",
            label_visibility="collapsed",
        )
        extraction_conf = "high"

    # ── IMAGE (upload / camera) ──
    elif mode == "Image":
        img_source = st.radio(
            "Image source",
            ["Upload file", "Capture from camera"],
            horizontal=True,
            key="img_src",
        )
        uploaded = None
        if img_source == "Upload file":
            uploaded = st.file_uploader(
                "Upload an image of a math problem",
                type=["png", "jpg", "jpeg", "webp"],
            )
        else:
            uploaded = st.camera_input("Take a photo of the math problem")

        if uploaded:
            st.image(uploaded, caption="Captured image", width="stretch")
            with st.spinner("Running OCR …"):
                raw_text, extraction_conf = extract_text_from_image(uploaded)
            _show_confidence("OCR", extraction_conf)

    # ── AUDIO (upload / record) ──
    elif mode == "Audio":
        audio_source = st.radio(
            "Audio source",
            ["Upload file", "Record audio"],
            horizontal=True,
            key="audio_src",
        )

        if audio_source == "Upload file":
            uploaded = st.file_uploader(
                "Upload an audio recording of a math problem",
                type=["wav", "mp3", "m4a", "ogg"],
            )
            if uploaded:
                st.audio(uploaded)
                with st.spinner("Transcribing audio …"):
                    raw_text, extraction_conf = extract_text_from_audio(uploaded)
                _show_confidence("Transcription", extraction_conf)
        else:
            try:
                from audio_recorder_streamlit import audio_recorder

                st.write("Click the microphone to start recording. Click again to stop.")
                audio_bytes = audio_recorder(
                    pause_threshold=3.0,
                    sample_rate=44100,
                    text="",
                    recording_color="#e74c3c",
                    neutral_color="#6c757d",
                )
                if audio_bytes:
                    st.audio(audio_bytes, format="audio/wav")
                    with st.spinner("Transcribing recorded audio …"):
                        raw_text, extraction_conf = extract_text_from_audio(BytesIO(audio_bytes))
                    _show_confidence("Transcription", extraction_conf)
            except ImportError:
                st.error(
                    "Audio recorder not installed. "
                    "Run `pip install audio-recorder-streamlit` and restart."
                )

    # Sync to session state
    if raw_text:
        st.session_state.extracted_text = raw_text
        st.session_state.extraction_confidence = extraction_conf
        st.session_state.input_mode = mode

    return raw_text, extraction_conf


def _show_confidence(label: str, conf: str | None):
    if conf:
        color = {"high": "green", "medium": "orange", "low": "red"}.get(conf, "gray")
        st.markdown(f"{label} confidence: :{color}[**{conf}**]")


# ────────────────────────────────────────────────────────────────
# HITL forms
# ────────────────────────────────────────────────────────────────

def render_hitl_low_confidence():
    st.warning(
        "**Low OCR / transcription confidence.**  "
        "Please review and correct the extracted text below."
    )
    corrected = st.text_area(
        "Correct the extracted text:",
        value=st.session_state.extracted_text,
        height=120,
        key="low_conf_correction",
    )
    if st.button("Confirm and Continue", type="primary"):
        st.session_state.extracted_text = corrected
        st.session_state.extraction_confidence = "high"
        st.session_state.hitl_stage = None
        run_pipeline(corrected)
        st.rerun()


def render_hitl_clarification():
    st.warning("**The problem seems ambiguous or incomplete. Please clarify.**")
    parsed = st.session_state.parsed_json
    st.json(parsed)
    clarified = st.text_area(
        "Edit or clarify your problem:",
        value=parsed.get("problem_text", st.session_state.extracted_text),
        key="clarification_input",
    )
    if st.button("Submit Clarification", type="primary"):
        st.session_state.parsed_json = None
        st.session_state.hitl_stage = None
        st.session_state.extracted_text = clarified
        run_pipeline(clarified)
        st.rerun()


def render_hitl_verification():
    st.warning("**The verifier is uncertain about this solution. Please review.**")
    parsed = st.session_state.parsed_json
    solver_res = st.session_state.solver_result
    verification = st.session_state.verification

    st.subheader("Problem")
    st.write(parsed["problem_text"])
    st.subheader("Proposed Solution")
    st.markdown(solver_res["solution"])
    st.subheader("Verifier Feedback")
    st.json(verification)

    action = st.radio(
        "Select an action",
        ["Approve as-is", "Edit solution", "Reject and re-solve"],
        key="hitl_action",
    )

    edited_sol = solver_res["solution"]
    if action == "Edit solution":
        edited_sol = st.text_area(
            "Edit the solution:", value=solver_res["solution"],
            height=200, key="edited_solution",
        )

    if st.button("Submit Review", type="primary"):
        if action in ("Approve as-is", "Edit solution"):
            st.session_state.solver_result["solution"] = edited_sol
            st.session_state.verification = {
                "is_correct": True,
                "confidence": "high",
                "feedback": "Approved by human reviewer.",
            }
            st.session_state.hitl_stage = None
            run_pipeline(parsed["problem_text"])
            st.rerun()
        else:
            st.session_state.solver_result = None
            st.session_state.verification = None
            st.session_state.hitl_stage = None
            run_pipeline(parsed["problem_text"])
            st.rerun()


# ────────────────────────────────────────────────────────────────
# Results display
# ────────────────────────────────────────────────────────────────

def _render_math(text: str):
    """Render markdown with LaTeX delimiters fixed for Streamlit."""
    # Convert \[ ... \] → $$ ... $$  (display math)
    result = re.sub(r'\\\[\s*', '\n$$\n', text)
    result = re.sub(r'\s*\\\]', '\n$$\n', result)
    # Convert \( ... \) → $ ... $  (inline math)
    result = re.sub(r'\\\(\s*', ' $', result)
    result = re.sub(r'\s*\\\)', '$ ', result)
    # Convert [ ... ] that contain LaTeX commands → $$ ... $$
    result = re.sub(r'\[\s*(.*?\\(?:frac|sqrt|pm|cdot|text|quad|int|sum|lim|infty).*?)\s*\]',
                    r'\n$$\1$$\n', result)
    st.markdown(result)


def render_results():
    # Confidence banner
    parsed = st.session_state.parsed_json
    solver_res = st.session_state.solver_result
    verification = st.session_state.verification
    explanation = st.session_state.explanation

    v_conf = verification.get("confidence", "medium")
    s_conf = solver_res.get("confidence", "medium")
    overall = "low" if "low" in (v_conf, s_conf) else ("medium" if "medium" in (v_conf, s_conf) else "high")

    conf_styles = {
        "high":   ("16a34a", "f0fdf4", "86efac", "Solved with high confidence"),
        "medium": ("d97706", "fffbeb", "fcd34d", "Solved — review recommended"),
        "low":    ("dc2626", "fef2f2", "fca5a5", "Low confidence — please verify"),
    }
    tc, bg, border, label = conf_styles.get(overall, conf_styles["medium"])
    st.markdown(
        f'<div style="background:#{bg};border:1px solid #{border};border-radius:10px;'
        f'padding:0.7rem 1.1rem;display:flex;align-items:center;gap:0.7rem;margin-bottom:1rem">'
        f'<span style="width:10px;height:10px;border-radius:50%;background:#{tc};flex-shrink:0"></span>'
        f'<span style="font-weight:700;color:#{tc};font-size:0.9rem">{label}</span>'
        f'<span style="color:#64748b;font-size:0.82rem;margin-left:auto">Overall: <strong>{overall.upper()}</strong></span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Tabs
    tab_explain, tab_solution, tab_context, tab_trace = st.tabs(
        ["Explanation", "Raw Solution", "Retrieved Context", "Agent Trace"]
    )

    with tab_explain:
        _render_math(explanation)

    with tab_solution:
        _render_math(solver_res["solution"])
        st.caption(f"Solver confidence: **{s_conf}**")

    with tab_context:
        st.markdown("**Knowledge Base Retrieval**")
        rag_ctx = st.session_state.rag_context
        if rag_ctx:
            for i, chunk in enumerate(rag_ctx.split("\n---\n"), 1):
                with st.expander(f"Source {i}", expanded=(i == 1)):
                    st.markdown(chunk)
        else:
            st.markdown(
                '<div style="color:#94a3b8;font-style:italic;font-size:0.88rem">'
                'No relevant knowledge-base entries retrieved.</div>',
                unsafe_allow_html=True,
            )
        st.markdown("**Memory — Similar Past Problems**")
        mem_ctx = st.session_state.memory_context
        if mem_ctx:
            st.markdown(mem_ctx)
        else:
            st.markdown(
                '<div style="color:#94a3b8;font-style:italic;font-size:0.88rem">'
                'No similar previously solved problems in memory.</div>',
                unsafe_allow_html=True,
            )

    with tab_trace:
        st.markdown("**Agent Execution Trace**")
        if st.session_state.agent_trace:
            for entry in st.session_state.agent_trace:
                st.markdown(
                    f"**`{entry['timestamp']}`** &nbsp;·&nbsp; **{entry['agent']}**\n\n"
                    f"&nbsp;&nbsp;Reason: {entry['reason']}\n\n"
                    f"&nbsp;&nbsp;Outcome: {entry['outcome']}"
                )
                st.divider()
        else:
            st.caption("No agent trace recorded.")
        with st.expander("Parser Output (JSON)"):
            st.json(parsed)
        with st.expander("Verifier Output (JSON)"):
            st.json(verification)

    # Feedback strip
    st.divider()
    st.markdown(
        '<p style="font-size:0.72rem;font-weight:700;text-transform:uppercase;'
        'letter-spacing:0.08em;color:#64748b;margin-bottom:0.6rem">Was this solution correct?</p>',
        unsafe_allow_html=True,
    )
    fb_col1, fb_col2, fb_col3 = st.columns([1, 1, 2])
    with fb_col1:
        if st.button("✓  Correct", key="fb_correct"):
            _save_feedback("correct", "")
            st.success("Feedback saved.")
    with fb_col2:
        if st.button("✗  Incorrect", key="fb_incorrect_btn"):
            st.session_state["_show_feedback_form"] = True

    if st.session_state.get("_show_feedback_form"):
        comment = st.text_area("What went wrong?", key="fb_comment", height=80)
        if st.button("Submit feedback", key="fb_submit"):
            _save_feedback("incorrect", comment)
            st.session_state["_show_feedback_form"] = False
            st.success("Feedback saved.")

    with fb_col3:
        if st.button("Request Re-check", key="recheck_btn"):
            st.session_state.solver_result = None
            st.session_state.verification = None
            st.session_state.explanation = None
            st.session_state.pipeline_complete = False
            st.session_state.hitl_stage = None
            st.session_state.agent_trace = []
            log_agent("User", "Explicitly requested re-check", "Re-running pipeline")
            run_pipeline(parsed["problem_text"])
            st.rerun()


def _save_feedback(label: str, comment: str):
    parsed = st.session_state.parsed_json
    if not parsed:
        return
    memory_save({
        "original_input": st.session_state.extracted_text,
        "input_mode": st.session_state.input_mode or "text",
        "parsed_question": parsed["problem_text"],
        "retrieved_context": st.session_state.rag_context,
        "final_answer": st.session_state.explanation,
        "verifier_outcome": st.session_state.verification,
        "user_feedback": {"label": label, "comment": comment},
    })
