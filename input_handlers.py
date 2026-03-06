"""
Multimodal input helpers — Math-aware OCR & ASR.

Primary:  OpenAI Vision (GPT-4o-mini) / Whisper API  (excellent math understanding).
Fallback: EasyOCR (images).
"""

import base64
import os
import re
import tempfile


# ── Prompts tuned for mathematical content ───────────────────────

_MATH_OCR_PROMPT = (
    "You are a specialist at reading mathematical notation from images.\n"
    "Extract the COMPLETE math problem exactly as written in the image.\n"
    "Rules:\n"
    "- Use ^ for exponents  (e.g. x^2)\n"
    "- Use √( ) for square roots\n"
    "- Write fractions as (a)/(b) when inline\n"
    "- Preserve ALL variables, constants, operators, and any surrounding text\n"
    "- For matrices use row notation: [a, b; c, d]\n"
    "- For integrals/summations keep limits: ∫_0^1  or  Σ_{i=1}^{n}\n"
    "- If multiple problems exist, separate them with blank lines\n"
    "Return ONLY the extracted math text.  No commentary or explanation."
)

_MATH_ASR_PROMPT = (
    "You are a specialist at transcribing spoken mathematics.\n"
    "Listen to this audio and convert it into precise mathematical notation.\n"
    "Conversion rules:\n"
    '  "x squared"          →  x^2\n'
    '  "x cubed"            →  x^3\n'
    '  "square root of x"   →  √(x)\n'
    '  "x to the power n"   →  x^n\n'
    '  "integral of f dx"   →  ∫ f dx\n'
    '  "derivative of f"    →  d/dx(f)\n'
    '  "limit as x approaches a" → lim_{x→a}\n'
    '  "summation"          →  Σ\n'
    '  "pi" → π,  "theta" → θ,  "alpha" → α,  "beta" → β\n'
    '  "infinity" → ∞\n'
    "Return ONLY the math problem.  No commentary."
)


# ── Helpers ──────────────────────────────────────────────────────

def _mime_type(name: str, fallback: str = "application/octet-stream") -> str:
    ext = os.path.splitext(name or "")[-1].lower()
    return {
        ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".webp": "image/webp", ".gif": "image/gif",
        ".wav": "audio/wav", ".mp3": "audio/mpeg",
        ".m4a": "audio/mp4", ".ogg": "audio/ogg",
    }.get(ext, fallback)


# ── Pix2Tex singleton (lazy-loaded, heavy model) ─────────────────
_pix2tex_model = None


def _get_pix2tex():
    """Lazy-load and cache the Pix2Tex LatexOCR model."""
    global _pix2tex_model
    if _pix2tex_model is None:
        from pix2tex.cli import LatexOCR
        _pix2tex_model = LatexOCR()
    return _pix2tex_model


# ── IMAGE OCR ────────────────────────────────────────────────────

def extract_text_from_image(uploaded_file) -> tuple[str, str]:
    """
    Extract math from an image.
    1st choice: Pix2Tex (LaTeX OCR — offline, specialised for math).
    2nd choice: OpenAI Vision (if Pix2Tex confidence is low or fails).
    3rd choice: EasyOCR (final fallback).
    """
    img_bytes = uploaded_file.getvalue()

    # ── 1. Pix2Tex (LaTeX OCR) ──
    pix2tex_error = None
    try:
        from PIL import Image
        from io import BytesIO as _BytesIO

        img = Image.open(_BytesIO(img_bytes))
        model = _get_pix2tex()
        latex = model(img)

        if latex and latex.strip():
            latex = latex.strip()
            # Heuristic confidence: if the output has common LaTeX commands
            # and isn't too short, consider it high confidence
            has_latex = any(cmd in latex for cmd in ("\\frac", "\\sqrt", "^", "_", "=", "+", "-"))
            is_reasonable_length = len(latex) > 5
            if has_latex and is_reasonable_length:
                # Wrap in $$ for display if not already wrapped
                display = f"$${latex}$$" if not latex.startswith("$") else latex
                return (display, "high")
            else:
                pix2tex_error = f"Low confidence output: {latex[:80]}"
    except ImportError:
        pix2tex_error = "pix2tex not installed"
    except Exception as exc:
        pix2tex_error = f"Pix2Tex error: {exc}"

    # ── 2. OpenAI Vision (LLM fallback) ──
    openai_error = None
    try:
        from config import get_openai_client, OPENAI_MODEL

        client = get_openai_client()
        if client is None:
            raise RuntimeError("OpenAI client not initialised (no API key yet)")

        b64_img = base64.b64encode(img_bytes).decode("utf-8")
        name = getattr(uploaded_file, "name", "image.png")
        mime = _mime_type(name, "image/png")

        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": _MATH_OCR_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime};base64,{b64_img}",
                                "detail": "high",
                            },
                        },
                    ],
                },
            ],
            temperature=0.1,
            max_tokens=1024,
        )
        text = response.choices[0].message.content
        if text and text.strip():
            return (text.strip(), "high")
        openai_error = "OpenAI returned empty text"
    except ImportError as exc:
        openai_error = f"Import error: {exc}"
    except Exception as exc:
        openai_error = f"OpenAI Vision error: {exc}"

    # Log fallback reasons
    try:
        import streamlit as _st
        _st.info(
            f"ℹ️ Using EasyOCR fallback "
            f"(Pix2Tex: {pix2tex_error}; OpenAI: {openai_error})"
        )
    except Exception:
        pass

    # ── 3. EasyOCR (final fallback) ──
    try:
        import easyocr

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(img_bytes)
            tmp_path = tmp.name

        reader = easyocr.Reader(["en"], gpu=False)
        results = reader.readtext(tmp_path, detail=1)
        os.unlink(tmp_path)

        if not results:
            return ("", "low")

        texts = [r[1] for r in results]
        scores = [r[2] for r in results]
        avg_conf = sum(scores) / len(scores) if scores else 0
        confidence = "high" if avg_conf >= 0.8 else "medium" if avg_conf >= 0.5 else "low"

        raw = " ".join(texts)
        return (_normalize_ocr_artefacts(raw), confidence)
    except ImportError:
        return ("[OCR unavailable] Please type the problem manually.", "low")


# ── AUDIO ASR ────────────────────────────────────────────────────

def extract_text_from_audio(uploaded_file) -> tuple[str, str]:
    """
    Transcribe audio of a math problem.
    Primary:  OpenAI Whisper API (cloud, no ffmpeg needed).
    Post-processing: GPT-4o-mini converts spoken math → symbolic notation.
    """
    audio_bytes = uploaded_file.getvalue()
    name = getattr(uploaded_file, "name", "audio.wav")
    suffix = os.path.splitext(name)[-1] or ".wav"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        from config import get_openai_client, OPENAI_MODEL

        client = get_openai_client()
        if client is None:
            return ("[Error] OpenAI client not initialised — enter your API key.", "low")

        # ── 1. Whisper API transcription ──
        with open(tmp_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                prompt="Math problem: equations, integrals, derivatives, x squared, pi, theta",
            )

        raw_text = transcription.text
        if not raw_text or not raw_text.strip():
            return ("[Transcription returned empty text]", "low")

        # ── 2. Post-process with GPT to convert spoken math → symbols ──
        try:
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": _MATH_ASR_PROMPT},
                    {"role": "user", "content": raw_text},
                ],
                temperature=0.1,
                max_tokens=512,
            )
            refined = response.choices[0].message.content
            if refined and refined.strip():
                return (refined.strip(), "high")
        except Exception:
            pass  # fall through to basic normalisation

        return (_normalize_math_phrases(raw_text.strip()), "medium")

    except ImportError:
        return ("[OpenAI not installed] Please type the problem manually.", "low")
    except Exception as exc:
        return (f"[Audio error] {exc}", "low")
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ── Post-processing for spoken math ─────────────────────────────

_SPOKEN_MATH = [
    # Longer phrases first to avoid partial matches
    ("square root of", "√"),
    ("cube root of", "∛"),
    ("raised to the power of", "^"),
    ("to the power of", "^"),
    ("raised to", "^"),
    ("divided by", "/"),
    ("multiplied by", "×"),
    ("x squared", "x²"),
    ("x cubed", "x³"),
    ("y squared", "y²"),
    ("a squared", "a²"),
    ("b squared", "b²"),
    ("n squared", "n²"),
    ("integral of", "∫"),
    ("derivative of", "d/dx"),
    ("summation of", "Σ"),
    ("limit as", "lim"),
    ("approaches", "→"),
    ("tends to", "→"),
    ("less than or equal to", "≤"),
    ("greater than or equal to", "≥"),
    ("not equal to", "≠"),
    ("less than", "<"),
    ("greater than", ">"),
    ("times", "×"),
    ("plus or minus", "±"),
    ("plus", "+"),
    ("minus", "-"),
    ("equals", "="),
    ("pi", "π"),
    ("theta", "θ"),
    ("alpha", "α"),
    ("beta", "β"),
    ("gamma", "γ"),
    ("delta", "δ"),
    ("epsilon", "ε"),
    ("lambda", "λ"),
    ("sigma", "σ"),
    ("omega", "ω"),
    ("infinity", "∞"),
    ("belongs to", "∈"),
    ("subset of", "⊂"),
    ("union", "∪"),
    ("intersection", "∩"),
    ("for all", "∀"),
    ("there exists", "∃"),
    ("perpendicular", "⊥"),
    ("parallel to", "∥"),
    ("angle", "∠"),
    ("degree", "°"),
    ("degrees", "°"),
]


def _normalize_math_phrases(text: str) -> str:
    """Replace common spoken math phrases with symbolic equivalents."""
    result = text
    for phrase, symbol in _SPOKEN_MATH:
        result = result.replace(phrase, symbol)
    return result


def _normalize_ocr_artefacts(text: str) -> str:
    """Fix common OCR misreads in mathematical text."""
    fixes = [
        (r"\bl\b(?=\s*[=+\-*/^])", "1"),   # lone 'l' near operators → '1'
        (r"\bO\b(?=\s*[.]\d)", "0"),        # 'O' before decimal → '0'
        (r"(?<!\w)[xX](?=\s*[²³])", "x"),   # normalise x
    ]
    result = text
    for pattern, repl in fixes:
        result = re.sub(pattern, repl, result)
    return result
