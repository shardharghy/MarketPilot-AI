
"""
services/gemini_provider.py — Gemini text generation wrapper.

All agents use this shared provider.
Includes automatic retries for temporary Gemini API failures.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Dict, Optional

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)


class GeminiProvider:
    """
    Singleton Gemini text client with automatic retry support.
    """

    _instance: Optional[GeminiProvider] = None
    _client: Optional[genai.Client] = None

    DEFAULT_TEXT_MODEL = "gemini-3.5-flash-lite"

    MAX_RETRIES = 4
    INITIAL_RETRY_DELAY = 2

    def __new__(cls) -> GeminiProvider:
        if cls._instance is None:
            api_key = os.getenv("GEMINI_API_KEY")

            if not api_key:
                raise EnvironmentError(
                    "GEMINI_API_KEY not found. "
                    "Create a .env file with GEMINI_API_KEY=your_key_here."
                )

            cls._client = genai.Client(api_key=api_key)
            cls._instance = super().__new__(cls)

            logger.info(
                "GeminiProvider initialized. Model: %s",
                cls.DEFAULT_TEXT_MODEL,
            )

        return cls._instance

    @property
    def client(self) -> genai.Client:
        return self.__class__._client

    # ---------------------------------------------------------
    # Retry utilities
    # ---------------------------------------------------------

    @staticmethod
    def _is_retryable_error(exc: Exception) -> bool:
        """Identify temporary API errors that can be retried."""

        retryable_codes = {429, 500, 502, 503, 504}

        status = getattr(exc, "status_code", None)

        if status is None:
            status = getattr(exc, "code", None)

        try:
            if int(status) in retryable_codes:
                return True
        except (TypeError, ValueError):
            pass

        message = str(exc).lower()

        retryable_terms = (
            "429",
            "500",
            "502",
            "503",
            "504",
            "resource_exhausted",
            "unavailable",
            "internal",
            "deadline_exceeded",
            "temporarily unavailable",
            "high demand",
        )

        return any(term in message for term in retryable_terms)

    def _generate_content_with_retry(
        self,
        model: str,
        contents: str,
        config: types.GenerateContentConfig,
    ) -> Any:
        """Call Gemini with exponential backoff for temporary failures."""

        for attempt in range(self.MAX_RETRIES):

            try:
                return self.client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config,
                )

            except Exception as exc:

                final_attempt = attempt == self.MAX_RETRIES - 1

                if final_attempt or not self._is_retryable_error(exc):
                    logger.error(
                        "Gemini request failed after %d attempt(s): %s",
                        attempt + 1,
                        exc,
                    )
                    raise

                delay = self.INITIAL_RETRY_DELAY * (2 ** attempt)

                logger.warning(
                    "Temporary Gemini error. Attempt %d/%d. "
                    "Retrying in %d seconds. Error: %s",
                    attempt + 1,
                    self.MAX_RETRIES,
                    delay,
                    exc,
                )

                time.sleep(delay)

        raise RuntimeError("Gemini request failed unexpectedly.")

    # ---------------------------------------------------------
    # Text generation
    # ---------------------------------------------------------

    def generate_text(
        self,
        prompt: str,
        system_prompt: str = "",
        model: str = DEFAULT_TEXT_MODEL,
        temperature: float = 0.7,
    ) -> str:

        config = types.GenerateContentConfig(
            temperature=temperature,
            system_instruction=system_prompt if system_prompt else None,
        )

        response = self._generate_content_with_retry(
            model=model,
            contents=prompt,
            config=config,
        )

        if not response.text:
            raise ValueError("Gemini returned an empty text response.")

        return response.text

    # ---------------------------------------------------------
    # JSON generation
    # ---------------------------------------------------------

    def generate_json(
        self,
        prompt: str,
        system_prompt: str = "",
        model: str = DEFAULT_TEXT_MODEL,
        temperature: float = 0.3,
    ) -> Dict[str, Any]:

        full_system = (
            (system_prompt + "\n\n" if system_prompt else "")
            + "CRITICAL: Respond with ONLY valid JSON. "
            "No markdown, no code blocks, no extra text."
        )

        config = types.GenerateContentConfig(
            temperature=temperature,
            response_mime_type="application/json",
            system_instruction=full_system,
        )

        response = self._generate_content_with_retry(
            model=model,
            contents=prompt,
            config=config,
        )

        if not response.text:
            raise ValueError("Gemini returned an empty JSON response.")

        raw = self._strip_code_fences(response.text.strip())

        try:
            data = json.loads(raw)

            if not isinstance(data, dict):
                raise ValueError("Gemini JSON response must be an object.")

            return data

        except json.JSONDecodeError as exc:
            logger.error("Gemini returned invalid JSON: %s", raw)
            raise ValueError(
                f"Gemini returned invalid JSON: {exc}"
            ) from exc

    # ---------------------------------------------------------
    # Utilities
    # ---------------------------------------------------------

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        """Remove Markdown JSON code fences."""

        text = re.sub(
            r"^\s*```(?:json)?\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )

        text = re.sub(r"\s*```\s*$", "", text)

        return text.strip()

    @classmethod
    def reset_singleton(cls) -> None:
        """Reset the provider singleton."""

        cls._instance = None
        cls._client = None

    @staticmethod
    def list_available_models() -> list[str]:
        """List models accessible with the current API key."""

        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            return []

        client = genai.Client(api_key=api_key)

        return [model.name for model in client.models.list()]
