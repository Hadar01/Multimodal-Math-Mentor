"""
Centralised configuration — paths, model name, environment variables.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
KB_PATH = BASE_DIR / "knowledge_base.json"
DOCS_DIR = BASE_DIR / "docs"
CHROMA_DIR = BASE_DIR / ".chroma_data"

# ── API ────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-4o-mini"

# ── Free trial ─────────────────────────────────────────────────
FREE_TRIAL_LIMIT = 10  # solves using the owner's key before users must supply their own

# ── Shared OpenAI client (initialised lazily in app.py) ─────────
_openai_client = None

def get_openai_client():
    """Return the shared OpenAI client instance."""
    return _openai_client

def init_openai_client(api_key: str):
    """Create / replace the shared OpenAI client."""
    global _openai_client
    from openai import OpenAI
    _openai_client = OpenAI(api_key=api_key)
