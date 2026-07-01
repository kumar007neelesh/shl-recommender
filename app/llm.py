"""Minimal, dependency-light LLM client.

Supports OpenAI-compatible endpoints (Groq, OpenRouter) and Gemini's REST API via
httpx — no heavy vendor SDKs. The agent only ever asks for JSON, so the single
method here returns parsed JSON and tolerates fenced/garbage output. Provider
"none" (or a missing key) makes `available` False, and the agent switches to its
deterministic controller. Failures never crash the request."""
from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

import httpx

from .config import Settings

_OPENAI_COMPATIBLE_BASE = {
    "groq": "https://api.groq.com/openai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
}

_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.S)


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    m = _JSON_FENCE.search(text)
    if m:
        text = m.group(1).strip()
    # try direct, then first {...} blob
    for candidate in (text, _first_brace_blob(text)):
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except Exception:
            continue
    return None


def _first_brace_blob(text: str) -> Optional[str]:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


class LLMClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.provider = (settings.llm_provider or "none").lower()
        self.available = self.provider != "none" and bool(settings.llm_api_key)

    def complete_json(
        self, system: str, user: str, temperature: float = 0.0
    ) -> Optional[Dict[str, Any]]:
        if not self.available:
            return None
        for attempt in range(self.settings.llm_max_retries + 1):
            try:
                if self.provider in _OPENAI_COMPATIBLE_BASE:
                    raw = self._openai_compatible(system, user, temperature)
                elif self.provider == "gemini":
                    raw = self._gemini(system, user, temperature)
                else:
                    return None
                parsed = _extract_json(raw)
                if parsed is not None:
                    return parsed
            except Exception:
                if attempt >= self.settings.llm_max_retries:
                    return None
        return None

    def _openai_compatible(self, system: str, user: str, temperature: float) -> str:
        base = _OPENAI_COMPATIBLE_BASE[self.provider]
        payload = {
            "model": self.settings.llm_model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {"type": "json_object"},
        }
        headers = {"Authorization": f"Bearer {self.settings.llm_api_key}"}
        with httpx.Client(timeout=self.settings.llm_timeout_s) as client:
            r = client.post(f"{base}/chat/completions", json=payload, headers=headers)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]

    def _gemini(self, system: str, user: str, temperature: float) -> str:
        model = self.settings.llm_model or "gemini-2.5-flash"
        # Header auth is the current standard; avoids putting the key in the URL.
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent"
        )
        headers = {"x-goog-api-key": self.settings.llm_api_key}
        payload = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {
                "temperature": temperature,
                "responseMimeType": "application/json",
            },
        }
        with httpx.Client(timeout=self.settings.llm_timeout_s) as client:
            r = client.post(url, json=payload, headers=headers)
            r.raise_for_status()
            return r.json()["candidates"][0]["content"]["parts"][0]["text"]
