"""Tool-calling chat loop for the wafer agent."""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from wafer.agent.client import AmdClient
from wafer.agent.prompts import SYSTEM_PROMPT
from wafer.agent.session import AgentContext
from wafer.agent.tools import TOOL_DEFINITIONS, execute_tool


def _message_to_dict(msg) -> dict:
    """Normalize an OpenAI SDK message (or dict) to a plain messages[] entry."""
    if isinstance(msg, dict):
        return msg
    out: dict[str, Any] = {"role": msg.role, "content": msg.content}
    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls:
        out["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in tool_calls
        ]
    return out


def _usage_info(resp) -> dict | None:
    usage = getattr(resp, "usage", None)
    if usage is None:
        return None
    info = {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
        "reasoning_tokens": None,
    }
    details = getattr(usage, "completion_tokens_details", None)
    if details is not None:
        info["reasoning_tokens"] = getattr(details, "reasoning_tokens", None)
    return info


def _run_tools(tool_calls: list[dict], ctx: AgentContext) -> list[dict]:
    """Execute tool calls; run concurrently when there are two or more."""
    if len(tool_calls) == 1:
        tc = tool_calls[0]
        fn = tc["function"]
        t0 = time.perf_counter()
        result = execute_tool(fn["name"], fn.get("arguments") or "{}", ctx)
        return [{
            "tool": fn["name"],
            "arguments": fn.get("arguments") or "{}",
            "result": result,
            "ms": round((time.perf_counter() - t0) * 1000.0, 1),
            "tool_call_id": tc["id"],
        }]

    results: dict[str, dict] = {}

    def _one(tc: dict) -> dict:
        fn = tc["function"]
        t0 = time.perf_counter()
        result = execute_tool(fn["name"], fn.get("arguments") or "{}", ctx)
        return {
            "tool": fn["name"],
            "arguments": fn.get("arguments") or "{}",
            "result": result,
            "ms": round((time.perf_counter() - t0) * 1000.0, 1),
            "tool_call_id": tc["id"],
        }

    with ThreadPoolExecutor(max_workers=min(4, len(tool_calls))) as pool:
        futs = {pool.submit(_one, tc): tc["id"] for tc in tool_calls}
        for fut in as_completed(futs):
            item = fut.result()
            results[item["tool_call_id"]] = item
    # Preserve the model's tool_call order when appending messages
    return [results[tc["id"]] for tc in tool_calls]


def run_turn(client: AmdClient, ctx: AgentContext, user_message: str, *,
             max_rounds: int = 8, system_prompt: str = SYSTEM_PROMPT) -> dict:
    """Run one user question through the tool loop. Returns answer + tool_trace + timings."""
    t_total = time.perf_counter()
    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    tool_trace: list[dict] = []
    amd_calls: list[dict] = []
    answer = None
    finish_reason = None

    for round_i in range(max_rounds):
        try:
            # Keep max_tokens high enough for Qwen reasoning_tokens + a complete short answer.
            # (Thinking counts toward max_tokens; 512/768 often ends finish_reason=length.)
            t0 = time.perf_counter()
            resp = client.chat(messages, tools=TOOL_DEFINITIONS, max_tokens=1024)
            amd_ms = (time.perf_counter() - t0) * 1000.0
        except Exception as exc:
            return {
                "ok": False,
                "error": f"AMD API request failed: {exc}",
                "answer": None,
                "messages": messages,
                "tool_trace": tool_trace,
                "llm_model": client.config.model,
                "classifier": ctx.classifier_name,
                "as_of": ctx.as_of,
                "timings": {
                    "total_ms": round((time.perf_counter() - t_total) * 1000.0, 1),
                    "verify_model_ms": client.verify_model_ms,
                    "amd_calls": amd_calls,
                    "tools": [{"tool": t["tool"], "ms": t.get("ms")} for t in tool_trace],
                    "n_amd_rounds": len(amd_calls),
                    "n_tool_calls": len(tool_trace),
                },
            }

        meta = dict(getattr(client, "last_request_meta", {}) or {})
        choice = resp.choices[0]
        finish_reason = choice.finish_reason
        usage_info = _usage_info(resp)
        amd_calls.append({
            "round": round_i,
            "duration_ms": round(amd_ms, 1),
            "ms": round(amd_ms, 1),  # alias
            "finish_reason": finish_reason,
            "prompt_tokens": (usage_info or {}).get("prompt_tokens"),
            "completion_tokens": (usage_info or {}).get("completion_tokens"),
            "reasoning_tokens": (usage_info or {}).get("reasoning_tokens"),
            "total_tokens": (usage_info or {}).get("total_tokens"),
            "max_tokens": meta.get("max_tokens"),
            "reasoning_effort": meta.get("reasoning_effort"),
            "n_messages": meta.get("n_messages"),
            "usage": usage_info,
        })

        assistant = _message_to_dict(choice.message)
        messages.append(assistant)

        tool_calls = assistant.get("tool_calls") or []
        if not tool_calls:
            answer = assistant.get("content") or ""
            break

        executed = _run_tools(tool_calls, ctx)
        for item in executed:
            tool_trace.append(item)
            messages.append({
                "role": "tool",
                "tool_call_id": item["tool_call_id"],
                "content": item["result"],
            })
    else:
        answer = answer or ("Stopped after max tool rounds without a final answer. "
                            "Partial tool results are in tool_trace.")

    total_ms = (time.perf_counter() - t_total) * 1000.0
    amd_total = sum(c["duration_ms"] for c in amd_calls)
    tool_total = sum(t.get("ms", 0.0) for t in tool_trace)
    return {
        "ok": True,
        "answer": answer,
        "finish_reason": finish_reason,
        "messages": messages,
        "tool_trace": tool_trace,
        "llm_model": client.config.model,
        "classifier": ctx.classifier_name,
        "as_of": ctx.as_of,
        "timings": {
            "total_ms": round(total_ms, 1),
            "amd_total_ms": round(amd_total, 1),
            "tools_total_ms": round(tool_total, 1),
            "other_ms": round(total_ms - amd_total - tool_total, 1),
            "verify_model_ms": (round(client.verify_model_ms, 1)
                                if client.verify_model_ms is not None else None),
            "n_amd_rounds": len(amd_calls),
            "n_tool_calls": len(tool_trace),
            "amd_calls": amd_calls,
            "tools": [{"tool": t["tool"], "ms": t.get("ms")} for t in tool_trace],
        },
    }
