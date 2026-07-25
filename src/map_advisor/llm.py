"""LLM abstraction layer.

The orchestrator and specialists never call a model SDK directly — they go
through :class:`LLMClient`. This keeps the agents provider-agnostic and lets
the whole system run with zero external dependencies via the bundled
:class:`MockLLMClient`.

Two concrete clients ship out of the box:

- :class:`MockLLMClient` — deterministic, canned responses for tests and CLI
  demos. No network, no API key, no cost.
- :class:`OpenAIClient` — a thin adapter over the ``openai`` Python SDK that
  works against any OpenAI-compatible endpoint (OpenAI, Azure OpenAI, vLLM,
  LM Studio, Ollama's OpenAI shim, etc.). Imported lazily so the core package
  has no hard dependency on ``openai``.

A third convenience factory, :func:`make_client`, selects the right backend
from a string tag — useful for wiring the CLI or tests to different providers.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from .errors import GuardrailError

__all__ = [
    "LLMResponse",
    "LLMClient",
    "MockLLMClient",
    "OpenAIClient",
    "make_client",
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class LLMResponse:
    """A normalized response from any LLM backend.

    Attributes:
        text: The model's textual output.
        raw: The original backend-specific object (may be ``None`` for mocks).
        meta: Free-form metadata (tokens used, latency, model id, ...).
    """

    text: str
    raw: Any = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.text


# ---------------------------------------------------------------------------
# Base contract
# ---------------------------------------------------------------------------

class LLMClient:
    """Abstract base. Subclasses implement :meth:`generate`."""

    def generate(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        stop: Optional[Sequence[str]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> LLMResponse:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Mock client — the workhorse for tests and offline demos
# ---------------------------------------------------------------------------

class MockLLMClient(LLMClient):
    """A deterministic stub LLM.

    Two modes:

    1. **Scripted responses.** Pass a list of canned strings; each call pops
       the next one. Useful for asserting exact orchestrator behavior.
    2. **Pattern responses.** Pass a mapping of regex -> response string.
       The first matching pattern wins; if none match, fall back to a
       generic acknowledgment. This is what powers the CLI demo.

    The mock also lets tests register custom callbacks via
    :meth:`respond_with` for edge cases (e.g. a specialist that misbehaves).
    """

    _STOP_TAG = "::stop"

    def __init__(
        self,
        responses: Optional[Sequence[str]] = None,
        patterns: Optional[Mapping[str, str]] = None,
        default: Optional[str] = None,
    ) -> None:
        self._scripted: List[str] = list(responses) if responses else []
        self._patterns: Dict[str, str] = dict(patterns or {})
        self._callbacks: List[Callable[[Sequence[Mapping[str, str]], Mapping[str, Any]], str]] = []
        self._default = default or "[mock] I have no canned response for that."
        self._call_log: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ API

    def respond_with(self, fn: Callable[[Sequence[Mapping[str, str]], Mapping[str, Any]], str]) -> None:
        """Register a callback consulted before patterns/scripted responses."""
        self._callbacks.append(fn)
        return None

    @property
    def call_log(self) -> List[Dict[str, Any]]:
        """Read-only log of every ``generate()`` invocation."""
        return list(self._call_log)

    def generate(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        stop: Optional[Sequence[str]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> LLMResponse:
        meta = dict(metadata or {})
        meta.update({
            "system_prompt": system_prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stop": list(stop) if stop else None,
        })
        self._call_log.append({"messages": list(messages), "meta": meta})

        # 1) Registered callbacks, first-come-first-served.
        for fn in self._callbacks:
            out = fn(messages, meta)
            if out is not None:
                return LLMResponse(text=self._strip_stop(out), meta=meta)

        # 2) Scripted queue (FIFO).
        if self._scripted:
            out = self._scripted.pop(0)
            return LLMResponse(text=self._strip_stop(out), meta=meta)

        # 3) Regex pattern map keyed on the last user message.
        last_user = self._last_user_text(messages)
        for pattern, resp in self._patterns.items():
            if re.search(pattern, last_user, re.IGNORECASE):
                return LLMResponse(text=self._strip_stop(resp), meta=meta)

        # 4) Fallback.
        return LLMResponse(text=self._strip_stop(self._default), meta=meta)

    # ------------------------------------------------------------------ util

    @staticmethod
    def _last_user_text(messages: Sequence[Mapping[str, str]]) -> str:
        for m in reversed(messages):
            if m.get("role") == "user":
                return str(m.get("content", ""))
        return ""

    @staticmethod
    def _strip_stop(text: str) -> str:
        """If a canned response ends with the stop tag, strip it."""
        if text.endswith(MockLLMClient._STOP_TAG):
            return text[: -len(MockLLMClient._STOP_TAG)].rstrip()
        return text


# ---------------------------------------------------------------------------
# OpenAI-compatible client (optional dependency)
# ---------------------------------------------------------------------------

class OpenAIClient(LLMClient):
    """Thin adapter over the ``openai`` Python SDK.

    Compatible with any OpenAI-shaped endpoint. Configure via constructor;
    falls back to these environment variables if args are omitted:
    ``OPENAI_API_KEY``, ``OPENAI_BASE_URL``, ``OPENAI_MODEL``.

    The ``openai`` package is imported lazily inside :meth:`generate` so that
    importing this module never forces the dependency on users who only want
    the mock backend.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        client: Any = None,
    ) -> None:
        self.model = model
        self._api_key = api_key
        self._base_url = base_url
        self._client = client  # injectable for tests

    def generate(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        stop: Optional[Sequence[str]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> LLMResponse:
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:  # pragma: no cover - exercised only with openai absent
            raise ImportError(
                "OpenAIClient requires the 'openai' package. "
                "Install it with: pip install 'map-advisor[openai]'"
            ) from exc

        client = self._client or OpenAI(api_key=self._api_key, base_url=self._base_url)
        full_messages: List[Dict[str, str]] = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend({"role": m["role"], "content": m["content"]} for m in messages)

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": full_messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if stop:
            kwargs["stop"] = list(stop)

        completion = client.chat.completions.create(**kwargs)
        text = completion.choices[0].message.content or ""
        return LLMResponse(text=text, raw=completion, meta={"model": self.model})


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_client(kind: str = "mock", **kwargs: Any) -> LLMClient:
    """Select a backend by name.

    Args:
        kind: One of ``"mock"``, ``"openai"``.
        **kwargs: Forwarded to the chosen client constructor.

    For the ``mock`` backend you may pass ``patterns`` (a JSON string will be
    decoded) so the CLI can configure canned responses from the shell.
    """
    kind = kind.lower().strip()
    if kind == "mock":
        if "patterns" in kwargs and isinstance(kwargs["patterns"], str):
            kwargs["patterns"] = json.loads(kwargs["patterns"])
        if "responses" in kwargs and isinstance(kwargs["responses"], str):
            kwargs["responses"] = json.loads(kwargs["responses"])
        return MockLLMClient(**kwargs)
    if kind == "openai":
        return OpenAIClient(**kwargs)
    raise GuardrailError(f"Unknown LLM backend: {kind!r}")
