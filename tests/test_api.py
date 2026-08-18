"""
tests/test_api.py
-------------------
Tests app/main.py + app/api/routes.py using FastAPI's TestClient (in-process,
no real network calls needed - fast and deterministic).

NOTE: the app no longer auto-loads a sample dataset on startup (sample/demo
data was removed - the app now requires a real upload before analysis).
Tests that need an active dataset use the `client_with_data` fixture, which
uploads a small seed CSV first. `test_health_endpoint_starts_empty` and the
`test_*_before_upload_returns_400` tests specifically cover the new
empty-start behavior.
"""
import io

import pytest
from fastapi.testclient import TestClient

from app.data import store
from app.main import app

SEED_CSV = (
    "Date,Region,Product,Category,Customer_Segment,Quantity,Revenue,Cost,Profit\n"
    "2024-01-01,North,Product A,Electronics,Consumer,10,1000,600,400\n"
    "2024-01-02,North,Product A,Electronics,Consumer,12,1200,700,500\n"
    "2024-01-01,West,Product B,Electronics,Consumer,8,800,500,300\n"
    "2024-01-02,West,Product B,Electronics,Consumer,9,900,550,350\n"
).encode("utf-8")


@pytest.fixture
def client():
    # No sample data is auto-loaded, but the in-memory store is a
    # module-level singleton that otherwise persists across tests in the
    # same pytest process - reset it explicitly for test isolation (same
    # pattern tests/test_agent.py already uses for agent_core._sessions).
    store._current_df = None
    store._current_report = None
    store._current_source_name = None
    with TestClient(app) as c:
        yield c


@pytest.fixture
def client_with_data(client):
    """A TestClient that already has a small valid dataset uploaded - use
    this for tests that exercise /analyze or /ask."""
    files = {"file": ("seed.csv", io.BytesIO(SEED_CSV), "text/csv")}
    resp = client.post("/upload", files=files)
    assert resp.status_code == 200
    return client


def test_health_endpoint_starts_empty(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["dataset_loaded"] is False
    assert body["dataset_rows"] == 0


def test_health_endpoint_after_upload(client_with_data):
    resp = client_with_data.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["dataset_loaded"] is True
    assert body["dataset_rows"] > 0


def test_root_endpoint(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "/health" in resp.json()["endpoints"]


def test_analyze_before_upload_returns_400(client):
    resp = client.post("/analyze", json={"period_days": 30})
    assert resp.status_code == 400


def test_ask_before_upload_returns_400(client):
    resp = client.post("/ask", json={"question": "Why did sales drop?"})
    assert resp.status_code == 400


def test_analyze_endpoint(client_with_data):
    resp = client_with_data.post("/analyze", json={"period_days": 30})
    assert resp.status_code == 200
    body = resp.json()
    assert "kpi_summary" in body
    assert "sales_analysis" in body
    assert "anomalies" in body
    assert "forecast" in body
    assert body["kpi_summary"]["total_revenue"] > 0


def test_analyze_endpoint_rejects_bad_period(client_with_data):
    resp = client_with_data.post("/analyze", json={"period_days": 1})  # below min (7)
    assert resp.status_code == 422  # pydantic validation error


def test_ask_endpoint(client_with_data):
    resp = client_with_data.post("/ask", json={"question": "Why did sales drop?", "session_id": "api-test"})
    assert resp.status_code == 200
    body = resp.json()
    assert "answer" in body
    assert "tool_calls" in body
    assert "analyze_sales" in body["tool_calls"]


def test_ask_endpoint_multiturn(client_with_data):
    resp1 = client_with_data.post("/ask", json={"question": "Why did sales drop?", "session_id": "api-multiturn"})
    assert resp1.status_code == 200
    resp2 = client_with_data.post("/ask", json={"question": "What about Region West?", "session_id": "api-multiturn"})
    assert resp2.status_code == 200
    assert resp2.json()["entities"]["region"] == "West"


def test_ask_endpoint_empty_question_rejected(client_with_data):
    resp = client_with_data.post("/ask", json={"question": ""})
    assert resp.status_code == 422


def test_upload_valid_csv(client):
    csv_content = (
        "Date,Region,Product,Category,Customer_Segment,Quantity,Revenue,Cost,Profit\n"
        "2024-01-01,North,Product A,Electronics,Consumer,10,1000,600,400\n"
        "2024-01-02,North,Product A,Electronics,Consumer,12,1200,700,500\n"
    ).encode("utf-8")
    files = {"file": ("test.csv", io.BytesIO(csv_content), "text/csv")}
    resp = client.post("/upload", files=files)
    assert resp.status_code == 200
    body = resp.json()
    assert body["rows_out"] == 2


def test_upload_missing_columns_returns_400(client):
    csv_content = b"Date,Region\n2024-01-01,North\n"
    files = {"file": ("bad.csv", io.BytesIO(csv_content), "text/csv")}
    resp = client.post("/upload", files=files)
    assert resp.status_code == 400
    assert "missing" in resp.json()["detail"].lower()


def test_upload_unsupported_extension_returns_400(client):
    files = {"file": ("data.txt", io.BytesIO(b"not a csv"), "text/plain")}
    resp = client.post("/upload", files=files)
    assert resp.status_code == 400
