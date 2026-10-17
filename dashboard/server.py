"""Yield Copilot dashboard (Phase 6): FastAPI backend + static single-page UI.

    pip install -r dashboard/requirements.txt
    uvicorn dashboard.server:app --port 8000          # from the repo root, then open http://localhost:8000

Replays a scored factory stream (Phase 4, default streams/stream_val.csv) as a live line. Every
number comes from the project's own modules: wafer.factory (alert rule, root causes, impact),
wafer.infer / wafer.explain (CNN result + attention heatmap), wafer.classifiers (CNN / YOLO) and
wafer.agent (LLM on AMD Token Factory). Nothing here re-implements their logic.

Environment:
    STREAM_CSV         scored stream to replay (default streams/stream_val_ens3.csv, scored by v0-ens3)
    WAFER_MODEL_DIR    model for the wafer inspector (default models/v0-ens3, the recommended model)
    AMD_API_KEY, AMD_BASE_URL, AMD_MODEL   enable the agent (see wafer/agent/config.py)
"""
from __future__ import annotations

import io
import json
import os
import threading
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from wafer.constants import CLASSES
from wafer.env import REPO_ROOT
from wafer.factory import check_alert, impact_from_alert, lookup_root_cause
from wafer.factory.history import stream_as_of
from wafer.factory.impact import WAFER_VALUE, WAFERS_PER_LOT

HERE = Path(__file__).resolve().parent
STREAM_CSV = Path(os.environ.get("STREAM_CSV") or next(
    p for p in (REPO_ROOT / "streams" / "stream_val_ens3.csv", REPO_ROOT / "streams" / "stream_val.csv")
    if p.exists()))
MODEL_DIR = Path(os.environ.get("WAFER_MODEL_DIR") or REPO_ROOT / "models" / "v0-ens3")
os.environ.setdefault("WAFER_MODEL_DIR", str(MODEL_DIR))   # the agent's CNN backend uses the same model
# Plain wafer map in the dashboard palette: off-wafer transparent, passing die grey, failing die crimson.
MAP_RGBA = np.array([[0, 0, 0, 0], [214, 218, 225, 255], [179, 18, 27, 255]], dtype=np.uint8)
REPORT_DIR = REPO_ROOT / "docs" / "results" / "report"

app = FastAPI(title="Yield Copilot", docs_url="/api/docs", redoc_url=None)
_lock = threading.Lock()
_decisions: list[dict] = []          # human-in-the-loop decisions (in memory for the demo)


# ----------------------------------------------------------------------------- data
@lru_cache(maxsize=1)
def stream() -> pd.DataFrame:
    df = pd.read_csv(STREAM_CSV)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if "pattern" not in df:
        raise RuntimeError(f"{STREAM_CSV} has no predictions; score it with wafer.factory.score_stream first")
    df["needs_review"] = df["needs_review"].astype(bool)
    df["review_reasons"] = df["review_reasons"].fillna("")
    return df.sort_values(["timestamp", "lot_id", "wafer_in_lot"]).reset_index(drop=True)


def wafer_map(uid: int) -> np.ndarray:
    """Raw map via wafer.data.access (demo stream pack first, then labeled.pkl if present)."""
    try:
        from wafer.data.access import get_wafer_map
        return get_wafer_map(uid)
    except FileNotFoundError as e:
        raise HTTPException(404, f"no map for uid {uid}: {e}") from e


# ----------------------------------------------------------------------------- models
@lru_cache(maxsize=1)
def cnn():
    from wafer.infer import load_model
    return load_model(MODEL_DIR, "cpu")


@lru_cache(maxsize=1)
def yolo():
    """(backend, None) or (None, reason). Never substitutes the CNN for YOLO."""
    try:
        from wafer.classifiers import get_classifier
        return get_classifier("yolo"), None
    except Exception as e:                                    # ClassifierNotAvailableError etc.
        return None, str(e) or type(e).__name__


_agent = None
_agent_error: str | None = None


def stream_classifier_name() -> str:
    """Backend that scored the loaded stream — agent history must match this."""
    df = stream()
    if "classifier" in df.columns and len(df):
        return str(df["classifier"].iloc[0])
    return (os.environ.get("WAFER_CLASSIFIER") or "cnn").strip().lower()


def agent():
    global _agent, _agent_error
    if _agent is None and _agent_error is None:
        try:
            from wafer.agent import WaferAgent
            from wafer.classifiers import get_classifier
            # Bind the agent to the stream's classifier so lot history / alerts match predictions.
            backend = get_classifier(stream_classifier_name())
            _agent = WaferAgent.from_env(
                stream(), as_of=str(stream()["timestamp"].iloc[0]),
                classifier=backend, verify_model=False,
            )
        except Exception as e:
            _agent_error = str(e).split(". Copy")[0]
    return _agent, _agent_error


# ----------------------------------------------------------------------------- helpers
def _iso(t) -> str:
    return pd.Timestamp(t).isoformat()


def _wafer_row(r) -> dict:
    return {"uid": int(r.uid), "lot_id": r.lot_id, "wafer": int(r.wafer_in_lot), "tool_id": r.tool_id,
            "timestamp": _iso(r.timestamp), "pattern": r.pattern, "confidence": float(r.confidence),
            "defect_prob": float(r.defect_prob), "needs_review": bool(r.needs_review),
            "review_reasons": [x for x in str(r.review_reasons).split(";") if x],
            "label": r.label, "model_version": r.model_version,
            "classifier": getattr(r, "classifier", None)}


def _decision_for(alert: dict) -> dict | None:
    for d in reversed(_decisions):
        if d["tool_id"] == alert["tool_id"] and d["fired_at"] == alert["fired_at"]:
            return d
    return None


# ----------------------------------------------------------------------------- API
@app.get("/api/meta")
def meta():
    df = stream()
    y_backend, yolo_err = yolo()
    ag, ag_err = agent()
    stream_clf = stream_classifier_name()
    yolo_meta = {
        "available": y_backend is not None,
        "reason": yolo_err,
        "experimental": True,
        "calibrated": False,
        "review_rules_validated": False,
        "model_version": "yolo-cls-exp",
        "disclosure": (
            "Experimental YOLO-cls; uncalibrated softmax; trained on a private stratified "
            "sample (not wafer_split_v1). Overlaps the val demo stream — not a fair "
            "comparison or held-out confirmation of the CNN."
        ),
    }
    stream_ver = str(df["model_version"].iloc[0]) if len(df) else ""
    line_calibrated = stream_clf == "cnn"
    return {
        "stream": STREAM_CSV.name, "n_wafers": len(df), "n_lots": int(df["lot_id"].nunique()),
        "tools": sorted(df["tool_id"].unique().tolist()),
        "timestamps": [_iso(t) for t in df["timestamp"].drop_duplicates()],
        "classes": CLASSES,
        "model": {
            "dir": MODEL_DIR.name,  # CNN weights dir (inspector / optional opinion)
            "stream_scored_by": f"{stream_clf} {stream_ver}".strip(),
            "stream_classifier": stream_clf,
            "stream_model_version": stream_ver,
            # Line model = predictions on the loaded stream (feed + alert rule).
            "line": {
                "classifier": stream_clf,
                "model_version": stream_ver,
                "drives_alerts": True,
                "calibrated": line_calibrated,
                "review_rules_validated": line_calibrated,
                "label": f"{stream_clf} {stream_ver}".strip(),
            },
        },
        "yolo": yolo_meta,
        "agent": {"available": ag is not None, "reason": ag_err,
                  "classifier": stream_clf if ag is not None else None},
        "wafer_value_usd": WAFER_VALUE, "wafers_per_lot": WAFERS_PER_LOT,
    }


@app.get("/api/state")
def state(as_of: str):
    df = stream()
    seen = stream_as_of(df, as_of)
    if seen.empty:
        return {"as_of": as_of, "kpis": None, "tools": [], "feed": [], "alerts": [], "decisions": _decisions}
    defect = seen["pattern"] != "none"
    alerts = check_alert(seen)
    out_alerts, exposure = [], 0.0
    for a in alerts:
        imp = impact_from_alert(seen, a, as_of=as_of)
        rc = lookup_root_cause(a["pattern"])
        v = imp.get("value_at_risk_usd") or 0.0
        exposure += v
        out_alerts.append({**a, "root_cause": {k: rc.get(k) for k in ("cause", "machine", "tool_family", "recommended_action", "source")},
                           "lots_since_alert": imp.get("lots_caught_early", 0), "wafers_since_alert": imp.get("wafers_at_risk") or 0,
                           "exposure_usd": v, "decision": _decision_for(a)})
    alert_tools = {a["tool_id"] for a in alerts}
    tools = []
    last24 = seen[seen["timestamp"] > pd.Timestamp(as_of) - pd.Timedelta(hours=24)]
    for t in sorted(df["tool_id"].unique()):
        g, g24 = seen[seen["tool_id"] == t], last24[last24["tool_id"] == t]
        per_lot = (g.assign(d=g["pattern"] != "none").groupby("lot_id", sort=False)["d"].sum())
        d24 = g24[g24["pattern"] != "none"]["pattern"].value_counts()
        tools.append({"tool_id": t, "status": "alert" if t in alert_tools else "ok", "wafers": int(len(g)),
                      "defect_rate": float((g["pattern"] != "none").mean()) if len(g) else 0.0,
                      "defects_24h": int(d24.sum()), "top_pattern_24h": d24.index[0] if len(d24) else None,
                      "per_lot_defects": [int(v) for v in per_lot.values], "per_lot_ids": list(per_lot.index),
                      "current_lot": g["lot_id"].iloc[-1] if len(g) else None})
    feed = [_wafer_row(r) for r in seen.tail(60).iloc[::-1].itertuples()]
    return {
        "as_of": as_of,
        "kpis": {"wafers": int(len(seen)), "lots": int(seen["lot_id"].nunique()),
                 "defect_rate": float(defect.mean()), "defects": int(defect.sum()),
                 "needs_review": int(seen["needs_review"].sum()), "alerts_open": sum(1 for a in out_alerts if not a["decision"]),
                 "alerts_total": len(out_alerts), "exposure_usd": exposure},
        "tools": tools, "feed": feed, "alerts": out_alerts, "decisions": list(reversed(_decisions))[:20],
    }


@app.get("/api/wafer/{uid}")
def wafer(uid: int):
    m = wafer_map(uid)
    row = stream()[stream()["uid"] == uid]
    ctx = _wafer_row(next(row.itertuples())) if len(row) else None
    stream_clf = stream_classifier_name()
    c = cnn().classify(m)
    y_backend, y_err = yolo()
    y_res = None
    if y_backend is not None:
        try:
            y_res = y_backend.classify(m)
        except Exception as e:
            y_err = str(e)
    # Predictions already on the stream drive the feed and alert rule.
    line = None
    if ctx and ctx.get("pattern") is not None:
        line = {
            "pattern": ctx["pattern"], "confidence": ctx["confidence"],
            "needs_review": ctx["needs_review"], "review_reasons": ctx["review_reasons"],
            "model_version": ctx["model_version"], "classifier": ctx.get("classifier") or stream_clf,
            "calibrated": stream_clf == "cnn",
            "review_rules_validated": stream_clf == "cnn",
            "drives_alerts": True,
        }
    line_pattern = line["pattern"] if line else None
    agree_cnn = (line_pattern == c["pattern"]) if line_pattern is not None else None
    agree_yolo = (line_pattern == y_res["pattern"]) if (line_pattern is not None and y_res) else None
    return {"uid": uid, "shape": list(m.shape), "dies": int((m > 0).sum()), "fails": int((m == 2).sum()),
            "line": line, "stream_classifier": stream_clf,
            "cnn": c, "yolo": y_res, "yolo_reason": y_err,
            "agree_cnn": agree_cnn, "agree_yolo": agree_yolo,
            # Back-compat: agree = line vs live YOLO when stream is CNN, else line vs CNN.
            "agree": agree_yolo if stream_clf == "cnn" else agree_cnn,
            "context": ctx}


@app.get("/api/wafer/{uid}/image.png")
def wafer_image(uid: int, attention: bool = False, size: int = 320):
    from PIL import Image

    from wafer.viz import render_heatmap, render_wafer
    m = wafer_map(uid)
    if attention:
        e = cnn().explain(m)
        img = render_heatmap(m, e["heatmap"], e["bbox"])
    else:
        img = MAP_RGBA[np.asarray(m, dtype=np.intp)]
    pil = Image.fromarray(img)
    scale = max(1, size // max(m.shape))
    pil = pil.resize((m.shape[1] * scale, m.shape[0] * scale), Image.NEAREST)
    buf = io.BytesIO()
    pil.save(buf, "PNG")
    return Response(buf.getvalue(), media_type="image/png", headers={"Cache-Control": "public, max-age=3600"})


class Decision(BaseModel):
    tool_id: str
    fired_at: str
    pattern: str
    action: str                     # hold | ticket | dismiss
    note: str = ""
    as_of: str


@app.post("/api/decisions")
def decide(d: Decision):
    if d.action not in ("hold", "ticket", "dismiss"):
        raise HTTPException(400, "action must be hold, ticket or dismiss")
    rec = {**d.model_dump(), "decided_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "by": "Engineer (demo)"}
    with _lock:
        _decisions.append(rec)
    return rec


class Question(BaseModel):
    question: str
    as_of: str


@app.post("/api/agent")
def ask(q: Question):
    ag, err = agent()
    if ag is None:
        return JSONResponse({"answer": None, "error": f"Agent not configured: {err}",
                             "as_of": q.as_of}, status_code=503)
    with _lock:
        ag.set_as_of(q.as_of)
        out = ag.ask(q.question)
    return {"answer": out.get("answer"), "error": out.get("error"), "as_of": q.as_of,
            "tools": [t.get("tool") for t in (out.get("timings") or {}).get("tools", [])]}


@app.get("/api/model")
def model_quality():
    name = MODEL_DIR.name
    f = REPO_ROOT / "docs" / "results" / f"{name}_val" / "metrics.json"
    if not f.exists():
        f = REPO_ROOT / "docs" / "results" / "v0-ens3_val" / "metrics.json"
    r = json.loads(f.read_text())
    charts = sorted(p.name for p in REPORT_DIR.glob("*.png")) if REPORT_DIR.exists() else []
    return {"model": f.parent.name.removesuffix("_val"), "split": r.get("split", "val"),
            "metrics": {k: r[k] for k in ("macro_f1", "defect_macro_f1", "accuracy", "balanced_accuracy",
                                          "defect_recall", "defect_precision", "ece", "review_rate")},
            "ci": r.get("ci", {}).get("macro_f1"), "per_class": r["per_class"], "charts": charts}


@app.get("/api/report/{name}")
def report_chart(name: str):
    p = (REPORT_DIR / name).resolve()
    if p.parent != REPORT_DIR.resolve() or p.suffix != ".png" or not p.exists():
        raise HTTPException(404, "chart not found")
    return FileResponse(p, media_type="image/png")


# ----------------------------------------------------------------------------- pages
# /         product homepage (dashboard/site)
# /signin   demo sign-in: no accounts, it only remembers a display name in the browser
# /app/     the dashboard itself (dashboard/static)
SITE = HERE / "site"


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(SITE / "index.html")


@app.get("/signin", include_in_schema=False)
def signin():
    return FileResponse(SITE / "signin.html")


@app.get("/app", include_in_schema=False)
def app_root(request: Request):
    q = request.url.query
    return RedirectResponse("/app/" + (f"?{q}" if q else ""))


app.mount("/site", StaticFiles(directory=SITE), name="site")
app.mount("/app", StaticFiles(directory=HERE / "static", html=True), name="app")
