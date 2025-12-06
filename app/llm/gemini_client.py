import os
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

GEMINI_API_KEY_ENV = "GEMINI_API_KEY"
GEMINI_MODEL_ENV = "GEMINI_MODEL_NAME"
DEFAULT_MODEL = os.getenv(GEMINI_MODEL_ENV, "gemini-2.5-flash-lite")


class GeminiClientError(RuntimeError):
    ...


@lru_cache(maxsize=1)
def _init_client(model_name: str = DEFAULT_MODEL):
    api_key = os.getenv(GEMINI_API_KEY_ENV)
    if not api_key:
        raise GeminiClientError(
            f"{GEMINI_API_KEY_ENV} is not set. Put it in .env or env vars."
        )
    if not model_name:
        raise GeminiClientError("No Gemini model name configured.")

    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)


def generate_text(
    prompt: str,
    model_name: str = DEFAULT_MODEL,
    system_instruction: Optional[str] = None,
    temperature: float = 0.2,
) -> str:
    model = _init_client(model_name)
    parts = []
    if system_instruction:
        parts.append(system_instruction)
    parts.append(prompt)
    resp = model.generate_content(
        parts,
        generation_config={"temperature": temperature},
    )
    return (resp.text or "").strip()
