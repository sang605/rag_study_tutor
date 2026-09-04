"""One small interface over several free LLM options.

Providers:
  gemini  - Google AI Studio free tier (needs GEMINI_API_KEY)
  groq    - Groq free tier (needs GROQ_API_KEY)
  ollama  - a model running locally on your own laptop, fully offline
  none    - no LLM at all: answers are stitched together from the retrieved text

`auto` picks the first one that is configured and falls back to `none`, so the
project always runs, even with an empty .env.
"""

from __future__ import annotations

from typing import Optional

import requests

from .config import Settings, settings as default_settings

TIMEOUT = 60


class LLMError(RuntimeError):
    pass


class BaseLLM:
    name = "base"
    generative = True

    def generate(self, prompt: str, system: str = "") -> str:
        raise NotImplementedError


class NoLLM(BaseLLM):
    """Extractive fallback - quotes your notes instead of writing new prose."""

    name = "none (extractive)"
    generative = False

    def generate(self, prompt: str, system: str = "") -> str:
        return (
            "No language model is configured, so here are the most relevant passages "
            "from your notes. Add a free GEMINI_API_KEY or GROQ_API_KEY to .env "
            "(or run Ollama) to get written answers."
        )


class GeminiLLM(BaseLLM):
    name = "gemini"

    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model

    def generate(self, prompt: str, system: str = "") -> str:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent"
        )
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        resp = requests.post(
            url,
            params={"key": self.api_key},
            json=payload,
            timeout=TIMEOUT,
        )
        if resp.status_code != 200:
            raise LLMError(f"Gemini error {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except (KeyError, IndexError) as exc:
            raise LLMError(f"Unexpected Gemini response: {data}") from exc


class GroqLLM(BaseLLM):
    name = "groq"

    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model

    def generate(self, prompt: str, system: str = "") -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "messages": messages, "temperature": 0.2},
            timeout=TIMEOUT,
        )
        if resp.status_code != 200:
            raise LLMError(f"Groq error {resp.status_code}: {resp.text[:300]}")
        return resp.json()["choices"][0]["message"]["content"].strip()


class OllamaLLM(BaseLLM):
    name = "ollama"

    def __init__(self, model: str, host: str):
        self.model = model
        self.host = host.rstrip("/")

    def generate(self, prompt: str, system: str = "") -> str:
        payload = {"model": self.model, "prompt": prompt, "stream": False}
        if system:
            payload["system"] = system
        try:
            resp = requests.post(
                f"{self.host}/api/generate", json=payload, timeout=TIMEOUT
            )
        except requests.RequestException as exc:
            raise LLMError(
                f"Could not reach Ollama at {self.host}. Is it running?"
            ) from exc
        if resp.status_code != 200:
            raise LLMError(f"Ollama error {resp.status_code}: {resp.text[:300]}")
        return resp.json().get("response", "").strip()


def _ollama_up(host: str) -> bool:
    try:
        return requests.get(f"{host.rstrip('/')}/api/tags", timeout=2).status_code == 200
    except requests.RequestException:
        return False


def get_llm(settings: Settings = default_settings) -> BaseLLM:
    provider = (settings.llm_provider or "auto").lower()

    if provider == "gemini":
        return GeminiLLM(settings.gemini_api_key, settings.gemini_model)
    if provider == "groq":
        return GroqLLM(settings.groq_api_key, settings.groq_model)
    if provider == "ollama":
        return OllamaLLM(settings.ollama_model, settings.ollama_host)
    if provider == "none":
        return NoLLM()

    # auto
    if settings.gemini_api_key:
        return GeminiLLM(settings.gemini_api_key, settings.gemini_model)
    if settings.groq_api_key:
        return GroqLLM(settings.groq_api_key, settings.groq_model)
    if _ollama_up(settings.ollama_host):
        return OllamaLLM(settings.ollama_model, settings.ollama_host)
    return NoLLM()
