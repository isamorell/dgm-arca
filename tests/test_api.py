"""The API must start with every phase pending and answer 501 with a useful message."""

from fastapi.testclient import TestClient

from api.app import app

client = TestClient(app)


def test_health_lists_all_phases():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert set(body["phases"]) == {"reasoning", "tools", "rag", "agent"}
    for phase in body["phases"].values():
        assert phase["status"] in {"ready", "pending", "error"}


def test_pending_phase_returns_501_with_phase_name():
    response = client.post("/tools", json={"query": "¿Qué tiempo hace en Madrid?"})
    if response.status_code == 501:
        assert response.json()["detail"]["phase"] == "tools"
    else:
        assert response.status_code == 200


def test_request_validation():
    assert client.post("/reasoning", json={}).status_code == 422
    assert client.post("/rag", json={"question": "x", "retriever": "magic"}).status_code == 422
    assert client.post("/agent", json={"task": "x", "max_steps": 0}).status_code == 422


def test_docs_available():
    assert client.get("/docs").status_code == 200
