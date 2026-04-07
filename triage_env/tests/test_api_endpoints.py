from fastapi.testclient import TestClient

from triage_env.server.app import app


client = TestClient(app)


def test_health_endpoint_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json().get("status") == "healthy"


def test_metadata_endpoint_ok():
    response = client.get("/metadata")
    assert response.status_code == 200
    body = response.json()
    assert "name" in body
    assert "version" in body


def test_schema_endpoint_ok():
    response = client.get("/schema")
    assert response.status_code == 200
    body = response.json()
    assert "action" in body


def test_state_endpoint_ok():
    response = client.get("/state")
    assert response.status_code == 200
    body = response.json()
    assert "episode_id" in body
    assert "step_count" in body


def test_reset_then_step_valid_action_returns_200():
    reset_response = client.post("/reset", json={})
    assert reset_response.status_code == 200

    step_response = client.post(
        "/step", json={"action": {"action_type": "wait", "patient_id": -1}}
    )
    assert step_response.status_code == 200
    body = step_response.json()
    assert "observation" in body


def test_step_invalid_action_returns_validation_error():
    client.post("/reset", json={})
    response = client.post(
        "/step", json={"action": {"action_type": "bad_action", "patient_id": 0}}
    )
    assert response.status_code == 422


def test_mcp_endpoint_returns_supported_error_payload():
    response = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
    )
    assert response.status_code == 200
    body = response.json()
    assert body.get("jsonrpc") == "2.0"
    assert "error" in body


def test_websocket_endpoint_handshake():
    with client.websocket_connect("/ws") as websocket:
        assert websocket is not None
