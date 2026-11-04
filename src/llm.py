"""A small client for Ollama's chat API, tuned for agent steps.

Agent steps run with thinking mode off: on qwen3:4b, thinking made structured
output unreliable (it returned empty JSON after ~600 tokens) and several times
slower. Structured steps pass a JSON schema in Ollama's ``format`` field.

Usage:
    llm = OllamaLLM()
    plan = llm.generate_json(system, user, schema=PLAN_SCHEMA)
"""

from __future__ import annotations

import json
import os
import re
import threading
import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen3:4b"
DEFAULT_TIMEOUT_SECONDS = 300.0  # the first request also loads the model into memory
# One context size for every call: Ollama reloads the model whenever num_ctx changes,
# which on an 8 GB machine can add minutes to the next call.
NUM_CTX = 6144
DEFAULT_MAX_TOKENS = 1024  # caps runaway output; small models occasionally loop until the timeout
DEFAULT_TEMPERATURE = 0.2
KEEP_ALIVE = "15m"  # keep the model loaded between agent steps

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_CLOSE = "</think>"


class LLMError(Exception):
    """The model could not produce a usable response."""


class LLMConnectionError(LLMError):
    """Ollama is not reachable."""


class LLMTimeoutError(LLMError):
    """Ollama is running but took too long, usually because the machine is short of memory.

    Not a connection error: Ollama is still up, so the run can continue without this one step.
    """


class LLM(Protocol):
    """What agents need from a language model. Tests provide a fake."""

    def generate(
        self, system: str, user: str, *, schema: dict[str, Any] | None = None, max_tokens: int = DEFAULT_MAX_TOKENS
    ) -> str: ...

    def generate_json(
        self, system: str, user: str, *, schema: dict[str, Any], max_tokens: int = DEFAULT_MAX_TOKENS
    ) -> dict[str, Any]: ...


@dataclass
class LLMStats:
    """Running totals for the calls made by one client."""

    calls: int = 0
    seconds: float = 0.0
    prompt_tokens: int = 0
    output_tokens: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, seconds: float, prompt_tokens: int, output_tokens: int) -> None:
        with self._lock:
            self.calls += 1
            self.seconds += seconds
            self.prompt_tokens += prompt_tokens
            self.output_tokens += output_tokens


class OllamaLLM:
    """Calls a local Ollama model with thinking off."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> None:
        self.model = model or os.getenv("OLLAMA_MODEL", DEFAULT_MODEL)
        self.base_url = (base_url or os.getenv("OLLAMA_URL", DEFAULT_OLLAMA_URL)).rstrip("/")
        self.timeout = timeout
        self.temperature = temperature
        self.stats = LLMStats()

    def generate(
        self, system: str, user: str, *, schema: dict[str, Any] | None = None, max_tokens: int = DEFAULT_MAX_TOKENS
    ) -> str:
        """Return the model's reply to one system + user message pair."""
        payload: dict[str, Any] = {
            "model": self.model,
            "stream": False,
            "think": False,
            "keep_alive": KEEP_ALIVE,
            "options": {"temperature": self.temperature, "num_ctx": NUM_CTX, "num_predict": max_tokens},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        if schema is not None:
            payload["format"] = schema

        started = time.monotonic()
        response = self._post("/api/chat", payload)
        message = response.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise LLMError("Ollama returned a response without message content.")
        self.stats.record(
            time.monotonic() - started,
            int(response.get("prompt_eval_count") or 0),
            int(response.get("eval_count") or 0),
        )
        if response.get("done_reason") == "length":
            logger.warning("Model output was cut off at %d tokens", max_tokens)
        return strip_reasoning(message["content"])

    def generate_json(
        self, system: str, user: str, *, schema: dict[str, Any], max_tokens: int = DEFAULT_MAX_TOKENS
    ) -> dict[str, Any]:
        """Return the model's reply parsed as a JSON object that follows ``schema``."""
        text = self.generate(system, user, schema=schema, max_tokens=max_tokens)
        return parse_json_object(text)

    def list_models(self) -> list[str]:
        """Return the names of downloaded models. Raises LLMConnectionError if Ollama is down."""
        response = self._request(urllib.request.Request(f"{self.base_url}/api/tags"))
        return [model["name"] for model in response.get("models", [])]

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        return self._request(request)

    def _request(self, request: urllib.request.Request) -> dict[str, Any]:
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            if exc.code == 404 and "not found" in detail.lower():
                raise LLMError(
                    f"Model '{self.model}' is not downloaded. Run: ollama pull {self.model}"
                ) from exc
            raise LLMError(f"Ollama returned HTTP {exc.code}: {detail}") from exc
        except (TimeoutError, urllib.error.URLError) as exc:
            reason = getattr(exc, "reason", exc)
            if isinstance(reason, TimeoutError) or isinstance(exc, TimeoutError):
                raise LLMTimeoutError(
                    f"Ollama did not respond within {self.timeout:.0f} seconds. "
                    "Closing other apps to free memory usually helps."
                ) from exc
            raise LLMConnectionError(
                f"Could not connect to Ollama at {self.base_url}. "
                "Make sure Ollama is running (open the Ollama app or run `ollama serve`)."
            ) from exc
        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise LLMError("Ollama returned invalid JSON.") from exc


def strip_reasoning(text: str) -> str:
    """Remove reasoning that qwen3 sometimes writes even with thinking off.

    It can appear as a full <think>...</think> block or as plain text ending in
    a lone </think> tag, so everything up to the last closing tag is dropped.
    """
    text = _THINK_BLOCK.sub("", text)
    close = text.lower().rfind(_THINK_CLOSE)
    if close != -1:
        text = text[close + len(_THINK_CLOSE) :]
    return text.strip()


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object from model output, tolerating surrounding text or code fences."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end < start:
        raise LLMError(f"Expected a JSON object from the model, got: {text[:200]!r}")
    try:
        value = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise LLMError(f"The model returned malformed JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise LLMError("Expected a JSON object from the model.")
    return value
