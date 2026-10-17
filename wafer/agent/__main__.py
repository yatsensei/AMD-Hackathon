"""Terminal entry: python -m wafer.agent --question '...' --stream ... --as-of ..."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from wafer.agent import WaferAgent
from wafer.classifiers import get_classifier
from wafer.classifiers.base import ClassifierNotAvailableError
from wafer.env import REPO_ROOT
from wafer.factory import score_stream


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Wafer defect pattern assistant (AMD Token Factory + CNN tools)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Requires AMD_API_KEY, AMD_BASE_URL, AMD_MODEL in the environment or .env "
            "(see .env.example). Never commit secrets.\n"
            "List models: GET $AMD_BASE_URL/models — set AMD_MODEL to a listed id.\n"
            "Classifier: cnn by default ($WAFER_CLASSIFIER). yolo is not available yet "
            "(clear error; no CNN fallback).\n"
            "Stdout is the answer. Errors and evidence warnings go to stderr. "
            "Use --debug for timings and the model/classifier/as_of footer.\n"
            "\n"
            "Works today: CNN classify + factory tools + Token Factory tool loop.\n"
            "Pending: YOLO adapter, Streamlit dashboard, final test-set eval.\n"
            "\n"
            "Score an unscored stream first if you need prediction-based alerts "
            "(needs data/interim/labeled.pkl + a CNN export):\n"
            "  python -m wafer.agent --score-stream streams/stream_val_unscored.csv "
            "-o streams/stream_val.csv\n"
            "\n"
            "Example:\n"
            "  python -m wafer.agent -q \"What is happening on ETCH-04?\" \\\n"
            "    --stream streams/stream_val.csv --as-of \"2026-10-13 04:10\""
        ),
    )
    p.add_argument("--question", "-q", help="User question for the agent")
    p.add_argument("--stream", default=str(REPO_ROOT / "streams" / "stream_val_unscored.csv"),
                   help="CSV path for the factory stream")
    p.add_argument("--as-of", default=None, help="Replay clock (default: last timestamp in stream)")
    p.add_argument("--classifier", default=None,
                   help="cnn (default) or yolo (experimental YOLO-cls; uncalibrated)")
    p.add_argument("--score-stream", metavar="PATH",
                   help="Score an unscored stream with the CNN and write CSV; then exit")
    p.add_argument("-o", "--out", default=None, help="Output path for --score-stream")
    p.add_argument("--max-rounds", type=int, default=8)
    p.add_argument("--no-verify-model", action="store_true",
                   help="Skip GET /v1/models check (offline / mocked)")
    p.add_argument("--json", action="store_true", help="Print full result JSON")
    p.add_argument("--debug", action="store_true",
                   help="Print timing diagnostics and model/classifier/as_of footer on stderr")
    a = p.parse_args(argv)

    if a.score_stream:
        try:
            backend = get_classifier(a.classifier or "cnn")
        except ClassifierNotAvailableError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        src = Path(a.score_stream)
        df = pd.read_csv(src)
        if "pattern" in df.columns:
            print(f"warning: {src} already has predictions; re-scoring", file=sys.stderr)
        scored = score_stream(df, classifier=backend)
        out = Path(a.out) if a.out else src.with_name(src.stem.replace("_unscored", "") + ".csv")
        if out == src and "_unscored" not in src.name:
            out = src.with_name(src.stem + "_scored.csv")
        out.parent.mkdir(parents=True, exist_ok=True)
        scored.to_csv(out, index=False)
        print(f"wrote {out} ({len(scored)} wafers, classifier={scored.classifier.iloc[0]}, "
              f"model_version={scored.model_version.iloc[0]})")
        return 0

    if not a.question:
        p.error("--question is required unless --score-stream is set")

    stream_path = Path(a.stream)
    if not stream_path.exists():
        print(f"error: stream not found: {stream_path}", file=sys.stderr)
        return 2
    stream_df = pd.read_csv(stream_path)
    as_of = a.as_of or str(pd.to_datetime(stream_df["timestamp"]).max())

    try:
        backend = get_classifier(a.classifier) if a.classifier else None
    except ClassifierNotAvailableError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        agent = WaferAgent.from_env(stream_df, as_of, classifier=backend,
                                    verify_model=not a.no_verify_model)
    except (EnvironmentError, ValueError, RuntimeError, ClassifierNotAvailableError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    result = agent.ask(a.question, max_rounds=a.max_rounds)
    if a.json:
        # timings included; messages omitted; never dump api keys
        print(json.dumps({k: v for k, v in result.items() if k != "messages"}, indent=2, default=str))
    else:
        if not result.get("ok"):
            print(f"error: {result.get('error')}", file=sys.stderr)
            return 1
        print(result.get("answer") or "")
        _print_evidence_warnings(result, file=sys.stderr)
        if a.debug:
            _print_timing_summary(result, file=sys.stderr)
    return 0 if result.get("ok", True) else 1


def _print_evidence_warnings(result: dict, file=sys.stderr) -> None:
    """Surface important evidence / completeness warnings (not timings)."""
    if result.get("finish_reason") == "length":
        print("warning: final answer may be truncated (finish_reason=length)", file=file)
    for item in result.get("tool_trace") or []:
        raw = item.get("result") or ""
        try:
            payload = json.loads(raw) if isinstance(raw, str) else raw
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        if payload.get("ok") is False and payload.get("error"):
            print(f"warning: tool {item.get('tool')}: {payload['error']}", file=file)
        note = payload.get("alerts_note") or payload.get("classifier_note")
        if note and payload.get("alerts_status") == "unavailable_no_predictions":
            print(f"warning: {note}", file=file)
        warnings = payload.get("source_warnings") or []
        if warnings and item.get("tool") == "lookup_root_cause":
            print(f"warning: root-cause source_warnings={warnings}", file=file)


def _print_timing_summary(result: dict, file=sys.stderr) -> None:
    """Human-readable timing breakdown (no secrets). Only used with --debug."""
    t = result.get("timings") or {}
    print(f"\n[model={result.get('llm_model')}; classifier={result.get('classifier')}; "
          f"as_of={result.get('as_of')}]", file=file)
    if t.get("verify_model_ms") is not None:
        print(f"  verify_model: {t['verify_model_ms']:.0f} ms", file=file)
    for i, call in enumerate(t.get("amd_calls") or []):
        print(
            f"  amd[{i}]: duration_ms={call.get('duration_ms', call.get('ms')):.0f}  "
            f"finish_reason={call.get('finish_reason')!r}  "
            f"prompt_tokens={call.get('prompt_tokens')}  "
            f"completion_tokens={call.get('completion_tokens')}  "
            f"reasoning_tokens={call.get('reasoning_tokens')}  "
            f"max_tokens={call.get('max_tokens')}  "
            f"reasoning_effort={call.get('reasoning_effort')!r}",
            file=file,
        )
    for tool in t.get("tools") or []:
        print(f"  tool {tool.get('tool')}: {tool.get('ms'):.0f} ms", file=file)
    if t:
        print(f"  rounds: amd={t.get('n_amd_rounds')}  tool_calls={t.get('n_tool_calls')}", file=file)
        print(f"  totals: amd={t.get('amd_total_ms'):.0f} ms  tools={t.get('tools_total_ms'):.0f} ms  "
              f"other={t.get('other_ms'):.0f} ms  wall={t.get('total_ms'):.0f} ms", file=file)


if __name__ == "__main__":
    raise SystemExit(main())
