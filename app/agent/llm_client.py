"""
app/agent/llm_client.py
------------------------
Thin wrapper around the Gemini API (Google's `google-genai` SDK). Isolated
in its own file so:
  1. The agent logic doesn't care which LLM provider is used.
  2. We have ONE place to implement the fallback behavior required by the
     challenge ("LLM fallback" in required features #12).

PROVIDER SWAP NOTE: this file originally wrapped Anthropic's API. It was
switched to Gemini per updated requirements. The public interface -
LLMClient.generate(system_prompt, user_prompt) -> str, LLMUnavailableError,
get_llm_client() - is UNCHANGED on purpose: app/agent/core.py only ever
talks to this interface, never to a vendor SDK directly, so the swap
required zero edits to core.py, tools.py, or anything else in the agent.

Fallback strategy:
  - If GEMINI_API_KEY is not set, LLM_PROVIDER != "gemini", the google-genai
    package isn't installed, or the API call fails/times out/returns empty,
    we fall back to a deterministic TEMPLATE-based narrative built directly
    from the tool results (see agent/core.py:_template_fallback). This
    means the app never crashes and never blocks on the LLM being down - it
    just gets less fluent, not less correct (numbers still come from the
    tools either way).
"""
from __future__ import annotations

from app.config import (
    GEMINI_API_KEY, GEMINI_MODEL, LLM_MAX_TOKENS,
    LLM_PROVIDER, LLM_TIMEOUT_SECONDS,
)
from app.logging_config import setup_logging

logger = setup_logging(__name__)


class LLMUnavailableError(Exception):
    """Raised when the LLM cannot be reached; callers should fall back."""


class LLMClient:
    def __init__(self):
        self._client = None
        self._types = None  # google.genai.types, stashed after a successful import
        self.enabled = bool(GEMINI_API_KEY) and LLM_PROVIDER == "gemini"

        if self.enabled:
            try:
                # Imported here (not at module level) so this whole module -
                # and everything that depends on it, i.e. the entire agent -
                # stays importable even in an environment where google-genai
                # isn't installed (mirrors how the Anthropic version isolated
                # its `import anthropic` the same way).
                from google import genai
                from google.genai import types

                self._types = types
                self._client = genai.Client(
                    api_key=GEMINI_API_KEY,
                    # google-genai's HttpOptions.timeout is in milliseconds.
                    http_options=types.HttpOptions(timeout=LLM_TIMEOUT_SECONDS * 1000),
                )
            except Exception:
                logger.exception("Failed to initialize Gemini client; disabling LLM.")
                self.enabled = False
        else:
            logger.warning(
                "LLM disabled (no GEMINI_API_KEY or LLM_PROVIDER != 'gemini'). "
                "Responses will use the deterministic template fallback."
            )

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Returns the LLM's text response. Raises LLMUnavailableError on any
        failure so the caller can apply the fallback."""
        if not self.enabled or self._client is None:
            raise LLMUnavailableError("LLM not configured (no API key).")

        try:
            response = self._client.models.generate_content(
                model=GEMINI_MODEL,
                contents=user_prompt,
                config=self._types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    max_output_tokens=LLM_MAX_TOKENS,
                ),
            )
            text = (response.text or "").strip()
            if not text:
                raise LLMUnavailableError("Gemini returned an empty response.")
            return text
        except LLMUnavailableError:
            raise
        except Exception as exc:
            logger.exception("LLM call failed")
            raise LLMUnavailableError(str(exc)) from exc


_llm_singleton: LLMClient | None = None


def get_llm_client() -> LLMClient:
    global _llm_singleton
    if _llm_singleton is None:
        _llm_singleton = LLMClient()
    return _llm_singleton
