"""Dashboard API: replay state, alert wiring, decisions, wafer images and the model page."""
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from dashboard import server  # noqa: E402

client = TestClient(server.app)


@pytest.fixture(scope="module")
def meta():
    r = client.get("/api/meta")
    assert r.status_code == 200
    return r.json()


def test_meta_describes_stream(meta):
    assert meta["n_wafers"] > 0 and meta["timestamps"]
    assert meta["tools"] and meta["classes"]
    assert meta["agent"]["available"] in (True, False)


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
