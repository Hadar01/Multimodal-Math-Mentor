"""
Pipeline orchestrator — runs the 5-agent sequence with HITL pauses.
"""

import streamlit as st

from agents import (
    log_agent,
    guardrail_agent,
    parser_agent,
    intent_router,
    solver_agent,
    verifier_agent,
    explainer_agent,
)
from memory import memory_save


# Session-state defaults used for reset
DEFAULTS = {
    "extracted_text": "",
    "extraction_confidence": None,
    "parsed_json": None,
    "solver_result": None,
    "rag_context": "",
    "memory_context": "",
    "verification": None,
    "explanation": None,
    "hitl_stage": None,
    "pipeline_complete": False,
    "agent_trace": [],
    "history": [],
    "input_mode": None,
    "free_tries_used": 0,
    "using_own_key": False,
    "guardrail_passed": False,
    "guardrail_blocked": None,
}


def init_session_state():
    """Ensure every expected key exists in st.session_state."""
    for key, default in DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default


def reset_pipeline():
    """Clear all pipeline state so the user can start a new problem."""
    for key in (
        "parsed_json", "solver_result", "verification",
        "explanation", "hitl_stage", "pipeline_complete",
        "rag_context", "memory_context", "agent_trace",
        "extraction_confidence", "input_mode",
        "guardrail_passed", "guardrail_blocked",
    ):
        st.session_state[key] = DEFAULTS[key]


def run_pipeline(text: str):
    """Execute the 5-agent pipeline, pausing at HITL checkpoints."""

    # ── Stage 0: Guardrail ──
    if st.session_state.get("guardrail_passed") is not True:
        with st.spinner("Checking input …"):
            result = guardrail_agent(text)
        if not result["allowed"]:
            st.session_state.guardrail_blocked = result["reason"]
            st.session_state.pipeline_complete = False
            return
        st.session_state.guardrail_passed = True
        st.session_state.guardrail_blocked = None
    # ── HITL: low OCR/ASR confidence ──
    conf = st.session_state.extraction_confidence
    if conf == "low" and st.session_state.hitl_stage is None:
        log_agent(
            "Input Validator",
            f"OCR/ASR confidence is '{conf}' → requesting human review",
            "HITL triggered",
        )
        st.session_state.hitl_stage = "low_confidence_input"
        return

    # ── Stage 1: Parser ──
    if st.session_state.parsed_json is None:
        with st.spinner("🔍 Parser Agent – analysing your problem …"):
            st.session_state.parsed_json = parser_agent(text)

    parsed = st.session_state.parsed_json

    # ── HITL: needs clarification ──
    route = intent_router(parsed)
    if route == "NEEDS_CLARIFICATION" and st.session_state.hitl_stage is None:
        st.session_state.hitl_stage = "clarification"
        return

    # ── Stage 2: Solver ──
    if st.session_state.solver_result is None:
        with st.spinner("🧠 Solver Agent – computing solution (with RAG + memory) …"):
            st.session_state.solver_result = solver_agent(parsed)

    solver_res = st.session_state.solver_result

    # ── Stage 3: Verifier ──
    if st.session_state.verification is None:
        with st.spinner("✅ Verifier Agent – checking correctness …"):
            st.session_state.verification = verifier_agent(parsed, solver_res)

    verification = st.session_state.verification

    # ── HITL: Verifier unsure / incorrect ──
    low_confidence = verification.get("confidence") == "low"
    incorrect = not verification.get("is_correct", True)
    if (low_confidence or incorrect) and st.session_state.hitl_stage is None:
        st.session_state.hitl_stage = "verification"
        return

    # ── Stage 4: Explainer ──
    if st.session_state.explanation is None:
        with st.spinner("📝 Explainer Agent – writing student-friendly explanation …"):
            st.session_state.explanation = explainer_agent(parsed, solver_res)

    # ── Save to Memory ──
    if not st.session_state.pipeline_complete:
        memory_save({
            "original_input": st.session_state.extracted_text,
            "input_mode": st.session_state.input_mode or "text",
            "parsed_question": parsed["problem_text"],
            "retrieved_context": st.session_state.rag_context,
            "final_answer": st.session_state.explanation,
            "verifier_outcome": verification,
            "user_feedback": None,
        })
        st.session_state.history.append({
            "question": parsed["problem_text"],
            "answer": st.session_state.explanation,
        })
        st.session_state.pipeline_complete = True
