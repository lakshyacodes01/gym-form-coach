import pytest

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _fake_rep(n=12):
    # 33 landmarks per frame as [x, y, z, visibility], spread out so torso > 0
    frame = [[0.5, 0.02 * i, 0.0, 1.0] for i in range(33)]
    return [frame for _ in range(n)]


def test_too_few_frames_is_unknown():
    # This path returns before the model loads, so it runs in CI without torch.
    resp = client.post("/api/predict", json={"frames": [[[0, 0, 0, 1]] * 33]})
    assert resp.status_code == 200
    assert resp.json()["label"] == "unknown"


def test_predict_returns_a_verdict():
    pytest.importorskip("torch")  # skips in CI (no torch); runs locally
    resp = client.post("/api/predict", json={"frames": _fake_rep()})
    assert resp.status_code == 200
    body = resp.json()
    assert body["label"] in ("good", "bad")
    assert 0.0 <= body["prob_good"] <= 1.0