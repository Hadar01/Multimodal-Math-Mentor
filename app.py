

import os
import warnings

# Suppress noisy third-party warnings from pix2tex dependencies
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
warnings.filterwarnings("ignore", category=UserWarning, module="albumentations")
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

import streamlit as st

from config import OPENAI_API_KEY, FREE_TRIAL_LIMIT, init_openai_client
from pipeline import init_session_state, reset_pipeline, run_pipeline
from ui import (
    apply_aiplanet_theme,
    render_page_hero,
    render_top_status_bar,
    render_sidebar,
    render_input,
    render_hitl_low_confidence,
    render_hitl_clarification,
    render_hitl_verification,
    render_results,
)

st.set_page_config(page_title="Multimodal Math Mentor", page_icon="🧮", layout="wide")
init_session_state()


def main():
    apply_aiplanet_theme()
    render_page_hero()
    render_top_status_bar()

    # ── Free-trial logic ──────────────────────────────────────
    free_left = FREE_TRIAL_LIMIT - st.session_state.get("free_tries_used", 0)
    trial_active = free_left > 0

    # Sidebar — API key + history
    api_key = render_sidebar(OPENAI_API_KEY, free_left=free_left, trial_active=trial_active)

    # Decide which key to actually use
    if trial_active and not st.session_state.get("using_own_key"):
        active_key = OPENAI_API_KEY          # owner's key for free trial
    else:
        active_key = api_key

    if not active_key:
        st.warning("You've used all 10 free solves. Enter your OpenAI API Key in the sidebar to continue.")
        st.stop()
    init_openai_client(active_key)

    # Input section
    _raw_text, extraction_conf = render_input()

    # Editable preview
    if st.session_state.extracted_text:
        st.markdown(
            '<p style="font-size:0.72rem;font-weight:700;text-transform:uppercase;'
            'letter-spacing:0.08em;color:#64748b;margin:0.8rem 0 0.3rem 0">Review Extracted Text</p>',
            unsafe_allow_html=True,
        )
        edited_text = st.text_area(
            "Extracted text",
            value=st.session_state.extracted_text,
            height=100,
            key="editable_text",
            label_visibility="collapsed",
        )
    else:
        edited_text = ""

    # Action buttons
    col1, col2 = st.columns([1, 4])
    with col1:
        solve_clicked = st.button("Solve", type="primary", disabled=not edited_text)
    with col2:
        if st.button("New Problem"):
            reset_pipeline()
            st.session_state.extracted_text = ""
            st.rerun()

    if solve_clicked:
        reset_pipeline()
        st.session_state.extracted_text = edited_text
        st.session_state.extraction_confidence = extraction_conf
        st.session_state.input_mode = st.session_state.get("input_mode")
        # Count against free trial only when using the owner key
        if trial_active and not st.session_state.get("using_own_key"):
            st.session_state.free_tries_used = st.session_state.get("free_tries_used", 0) + 1
        run_pipeline(edited_text)
        st.rerun()

    # Guardrail blocked
    if st.session_state.get("guardrail_blocked"):
        st.error(
            f"**Input blocked by safety guardrail.**\n\n"
            f"{st.session_state.guardrail_blocked}\n\n"
            f"Please rephrase your input as a math problem and try again."
        )

    # HITL pauses
    if st.session_state.hitl_stage == "low_confidence_input":
        render_hitl_low_confidence()
    elif st.session_state.hitl_stage == "clarification":
        render_hitl_clarification()
    elif st.session_state.hitl_stage == "verification":
        render_hitl_verification()

    # Results
    if st.session_state.pipeline_complete:
        render_results()


if __name__ == "__main__":
    main()
