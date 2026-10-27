"""Every endpoint: predictions, real characters, invalid input (422) and a missing model (503)."""

from app.main import MODELS
from app.schemas import EXAMPLE_CHARACTER

DOOMED = {
    "name": "Doomed Prince",
    "male": True,
    "house": "targaryen",
    "title": "royalty",
    "noble": True,
    "dead_relatives": 8,
    "popularity": 0.9,
    "age": 30,
    "books": [1],
}
SAFE = {"name": "Quiet Maester", "title": "maester", "books": [4]}


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "models": {"got": True}}


def test_models(client):
    [info] = client.get("/models").json()
    assert info["name"] == "got"
    assert info["loaded"] is True
    assert info["metrics"]["roc_auc"] > 0.75
    assert "book4" in info["importance"]


def test_predict_dies(client):
    response = client.post("/got/predict", json=DOOMED)
    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "Doomed Prince"
    assert body["verdict"] == "dies"
    assert body["dies"] is True
    assert body["death_probability"] >= body["threshold"]
    assert "Spoilers" in body["warning"]


def test_predict_survives(client):
    body = client.post("/got/predict", json=SAFE).json()
    assert body["verdict"] == "survives"
    assert body["survival_chance"].endswith("%")


def test_everything_is_optional(client):
    response = client.post("/got/predict", json={})
    assert response.status_code == 200
    assert response.json()["name"] == "Unnamed"


def test_example_works(client):
    assert client.post("/got/predict", json=EXAMPLE_CHARACTER).status_code == 200


def test_batch(client):
    response = client.post("/got/predict/batch", json=[DOOMED, SAFE])
    assert response.status_code == 200
    assert [p["verdict"] for p in response.json()] == ["dies", "survives"]
    assert client.post("/got/predict/batch", json=[]).json() == []


def test_real_character(client):
    [robb] = client.get("/got/characters", params={"search": "Robb Stark"}).json()
    assert robb["name"] == "Robb Stark"
    assert robb["books_say"] == "dead"
    assert robb["model_was_right"] == (robb["verdict"] == "dies")
    assert client.get("/got/characters", params={"search": "stark", "limit": 3}).json()
    assert len(client.get("/got/characters", params={"search": "stark", "limit": 3}).json()) == 3
    assert client.get("/got/characters", params={"search": "Sovann"}).json() == []


def test_invalid_input(client):
    assert client.post("/got/predict", json={"books": [6]}).status_code == 422
    assert client.post("/got/predict", json={"age": -5}).status_code == 422
    assert client.post("/got/predict", json={"house": "House Stark"}).status_code == 422
    assert client.post("/got/predict", json={"popularity": 2}).status_code == 422
    assert client.post("/got/predict", json={"has_dragons": True}).status_code == 422
    assert client.get("/got/characters", params={"search": "a"}).status_code == 422


def test_model_not_trained(client, monkeypatch):
    monkeypatch.setitem(MODELS, "got", None)
    response = client.post("/got/predict", json=DOOMED)
    assert response.status_code == 503
    assert "python train_got.py" in response.json()["detail"]


def test_unloadable_model_file(tmp_path, monkeypatch):
    """A broken or incompatible model file must not crash the API: it starts and answers 503."""
    import pytest
    from fastapi.testclient import TestClient

    from app import ml
    from app.main import app

    (tmp_path / "got.joblib").write_bytes(b"not a model")
    monkeypatch.setattr(ml, "MODELS_DIR", tmp_path)
    with pytest.warns(UserWarning, match="retrain"), TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok", "models": {"got": False}}
        assert client.post("/got/predict", json={}).status_code == 503
