"""OpenAI-compatible client for AMD Token Factory."""
from __future__ import annotations

import random
import time
from typing import Any

from wafer.agent.config import AgentConfig


def default_reasoning_effort(model_id: str) -> str | None:
    """Qwen shared models think by default; use low so answers aren't starved by thinking."""
    mid = (model_id or "").lower()
    if "qwen" in mid:
        return "low"
    return None


class AmdClient:
    """Thin wrapper: chat.completions with tools, model check, 429 backoff."""

    def __init__(self, config: AgentConfig, *, verify_model: bool = True, client=None):
        self.config = config
        self.last_request_ms: float | None = None
        self.verify_model_ms: float | None = None
        if client is not None:
            self._client = client
        else:
            from openai import OpenAI
            self._client = OpenAI(base_url=config.base_url, api_key=config.api_key)
        if verify_model and config.model:
            t0 = time.perf_counter()
            self._assert_model_available(config.model)
            self.verify_model_ms = (time.perf_counter() - t0) * 1000.0

    def _assert_model_available(self, model_id: str) -> None:
        try:
            listed = self._client.models.list()
            ids = {m.id for m in listed.data}
        except Exception as exc:  # network / auth — surface clearly
            raise RuntimeError(f"Could not list models at {self.config.base_url}: {exc}") from exc
        if model_id not in ids:
            sample = ", ".join(sorted(ids)[:12]) or "(none)"
            raise ValueError(
                f"AMD_MODEL {model_id!r} is not in GET /v1/models. "
                f"Available (sample): {sample}. Set AMD_MODEL to a listed id."
            )

    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             tool_choice: str | dict | None = "auto", max_tokens: int = 1024,
             temperature: float = 0.2, max_retries: int = 5,
             reasoning_effort: str | None = ...) -> Any:
        """Chat completion. `reasoning_effort=...` means auto (low for Qwen, else omit)."""
        if reasoning_effort is ...:
            reasoning_effort = default_reasoning_effort(self.config.model)

        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if reasoning_effort is not None:
            kwargs["reasoning_effort"] = reasoning_effort
        if tools:
            kwargs["tools"] = tools
            kwargs["parallel_tool_calls"] = True
            if tool_choice is not None:
                kwargs["tool_choice"] = tool_choice

        # Request metadata for timing reports (never includes api_key / messages body).
        self.last_request_meta = {
            "model": self.config.model,
            "max_tokens": max_tokens,
            "reasoning_effort": reasoning_effort,
            "parallel_tool_calls": bool(tools),
            "n_messages": len(messages),
            "n_tools": len(tools) if tools else 0,
        }

        delay = 1.0
        for attempt in range(max_retries + 1):
            t0 = time.perf_counter()
            try:
                resp = self._client.chat.completions.create(**kwargs)
                self.last_request_ms = (time.perf_counter() - t0) * 1000.0
                return resp
            except Exception as exc:
                self.last_request_ms = (time.perf_counter() - t0) * 1000.0
                status = getattr(exc, "status_code", None) or getattr(
                    getattr(exc, "response", None), "status_code", None)
                if status != 429 or attempt >= max_retries:
                    raise
                retry_after = None
                resp = getattr(exc, "response", None)
                if resp is not None:
                    retry_after = resp.headers.get("Retry-After") if hasattr(resp, "headers") else None
                wait = float(retry_after) if retry_after else delay + random.uniform(0, 0.5)
                time.sleep(wait)
                delay = min(delay * 2, 60.0)
