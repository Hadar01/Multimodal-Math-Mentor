"""
LLM helper — wraps the OpenAI Chat Completions API.

Every agent calls `call_llm()` so that all API logic lives in one place.
"""

from config import OPENAI_MODEL, get_openai_client


def call_llm(system_prompt: str, user_prompt: str) -> str:
    """Send a single system+user message pair to OpenAI and return the text."""
    client = get_openai_client()
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    text = response.choices[0].message.content
    if text:
        return text.strip()
    return "(Model returned no content — please rephrase the problem.)"
