"""Train the model once, into a temporary folder, for the whole test session."""

import pytest

import train_got
from app import ml


@pytest.fixture(scope="session")
def models_dir(tmp_path_factory):
    path = tmp_path_factory.mktemp("models")
    train_got.train(models_dir=path, verbose=False)
    return path


@pytest.fixture
def client(models_dir, monkeypatch):
    from fastapi.testclient import TestClient

    from app.main import app

    monkeypatch.setattr(ml, "MODELS_DIR", models_dir)
    with TestClient(app) as test_client:  # `with` runs the lifespan: the model is loaded
        yield test_client
