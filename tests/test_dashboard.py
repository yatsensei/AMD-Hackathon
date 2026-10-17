"""Dashboard API: replay state, alert wiring, decisions, wafer images and the model page."""
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from dashboard import server  # noqa: E402
from wafer.env import REPO_ROOT  # noqa: E402

client = TestClient(server.app)
STATIC = REPO_ROOT / "dashboard" / "static"


@pytest.fixture(scope="module")
def meta():
    r = client.get("/api/meta")
    assert r.status_code == 200
    return r.json()


def test_meta_describes_stream(meta):
    assert meta["n_wafers"] > 0 and meta["timestamps"]
    assert meta["tools"] and meta["classes"]
    assert meta["agent"]["available"] in (True, False)


def test_meta_line_model_separate_from_second_opinion(meta):
    """Line model drives feed/alerts; YOLO disclosure must not claim calibration."""
    line = meta["model"]["line"]
    assert line["drives_alerts"] is True
    assert line["classifier"] == meta["model"]["stream_classifier"]
    assert line["label"] == meta["model"]["stream_scored_by"]
    yolo = meta["yolo"]
    assert yolo["experimental"] is True
    assert yolo["calibrated"] is False
    assert yolo["review_rules_validated"] is False
    assert "uncalibrated" in yolo["disclosure"].lower()
    assert "overlap" in yolo["disclosure"].lower() or "overlaps" in yolo["disclosure"].lower()
    if line["classifier"] == "cnn":
        assert line["calibrated"] is True and line["review_rules_validated"] is True
    else:
        assert line["calibrated"] is False and line["review_rules_validated"] is False


def test_state_before_and_after_excursion(meta):
    early = client.get("/api/state", params={"as_of": meta["timestamps"][0]}).json()
    assert early["kpis"]["wafers"] >= 1 and early["alerts"] == []
    late = client.get("/api/state", params={"as_of": meta["timestamps"][-1]}).json()
    assert late["kpis"]["wafers"] == meta["n_wafers"]
    assert len(late["feed"]) <= 60 and {t["tool_id"] for t in late["tools"]} == set(meta["tools"])
    assert late["alerts"], "the val stream carries an injected excursion"
    a = late["alerts"][0]
    assert a["root_cause"]["cause"] and a["exposure_usd"] >= 0


def test_decision_closes_alert(meta):
    as_of = meta["timestamps"][-1]
    a = client.get("/api/state", params={"as_of": as_of}).json()["alerts"][0]
    r = client.post("/api/decisions", json={"tool_id": a["tool_id"], "fired_at": a["fired_at"],
                                             "pattern": a["pattern"], "action": "hold", "as_of": as_of})
    assert r.status_code == 200
    after = client.get("/api/state", params={"as_of": as_of}).json()
    assert after["alerts"][0]["decision"]["action"] == "hold"
    assert after["kpis"]["alerts_open"] == after["kpis"]["alerts_total"] - 1
    server._decisions.clear()


def test_wafer_image_is_png(meta):
    uid = client.get("/api/state", params={"as_of": meta["timestamps"][-1]}).json()["feed"][0]["uid"]
    r = client.get(f"/api/wafer/{uid}/image.png", params={"size": 64})
    assert r.status_code == 200 and r.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_model_page_and_report_guard():
    m = client.get("/api/model").json()
    assert 0 < m["metrics"]["macro_f1"] <= 1 and len(m["per_class"]) == 9
    assert client.get("/api/report/..%2F..%2FLICENSE").status_code == 404


def test_wafer_exposes_line_prediction(meta):
    uid = client.get("/api/state", params={"as_of": meta["timestamps"][-1]}).json()["feed"][0]["uid"]
    w = client.get(f"/api/wafer/{uid}").json()
    assert w["line"] is not None
    assert w["line"]["drives_alerts"] is True
    assert w["line"]["classifier"] == meta["model"]["stream_classifier"]
    assert "cnn" in w and ("yolo" in w or w.get("yolo_reason"))
    assert "agree_cnn" in w and "agree_yolo" in w


def test_agent_echoes_as_of(meta):
    """/api/agent must receive and return the replay timestamp from the request body."""
    as_of = meta["timestamps"][min(5, len(meta["timestamps"]) - 1)]
    r = client.post("/api/agent", json={"question": "ping", "as_of": as_of})
    body = r.json()
    assert body["as_of"] == as_of
    assert r.status_code in (200, 503)


def test_yolo_stream_meta_line_uncalibrated(monkeypatch):
    yolo_csv = REPO_ROOT / "streams" / "stream_val_yolo.csv"
    if not yolo_csv.exists():
        pytest.skip("stream_val_yolo.csv not present")
    monkeypatch.setattr(server, "STREAM_CSV", yolo_csv)
    server.stream.cache_clear()
    try:
        m = client.get("/api/meta").json()
        assert m["model"]["line"]["classifier"] == "yolo"
        assert m["model"]["line"]["calibrated"] is False
        assert m["model"]["line"]["review_rules_validated"] is False
        assert m["model"]["line"]["drives_alerts"] is True
        assert "Experimental" in m["yolo"]["disclosure"]
    finally:
        server.stream.cache_clear()


def test_ui_line_model_as_of_and_safe_markdown():
    js = (STATIC / "app.js").read_text()
    html = (STATIC / "index.html").read_text()
    assert "function renderMd" in js
    assert "esc(src)" in js  # Markdown rendered only after HTML escape
    assert 'body: JSON.stringify({ question: q, as_of: askedAt })' in js
    assert "Answer as of line time" in js
    assert "Uncalibrated" in js
    assert 'pill("#pill-model"' in js and "Line model" in js
    assert "Second opinion" in js
    assert 'id="steps-classifier"' in html
    assert "Line model" in html


def test_agent_prompt_yolo_and_sparse_evidence():
    from wafer.agent.prompts import SYSTEM_PROMPT
    low = SYSTEM_PROMPT.lower()
    assert "uncalibrated" in low
    assert "mixed or sparse" in low or "sparse pattern" in low
    assert "validated automatic acceptance" in low or "validated review policy" in low


def test_site_pages_and_app_route():
    home = client.get("/")
    assert home.status_code == 200 and "Catch the excursion" in home.text
    assert client.get("/signin").status_code == 200
    r = client.get("/app?t=330", follow_redirects=False)
    assert r.status_code == 307 and r.headers["location"] == "/app/?t=330"
    app_page = client.get("/app/")
    assert app_page.status_code == 200 and 'id="view-ops"' in app_page.text
    assert client.get("/app/app.js").status_code == 200


def test_site_assets_exist():
    import re
    html = client.get("/").text + client.get("/signin").text
    for src in set(re.findall(r'(?:src|href)="(/site/[^"]+)"', html)):
        assert client.get(src).status_code == 200, src
