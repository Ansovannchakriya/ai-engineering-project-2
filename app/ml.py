"""Load the trained model and make predictions with it (plain Python: no FastAPI here)."""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import joblib
import pandas as pd
import sklearn

# Where train_got.py saved the model (override for tests, or in Docker).
MODELS_DIR = Path(os.environ.get("MODELS_DIR", Path(__file__).parent.parent / "models"))

WARNING = "Spoilers! Trained on the books up to A Dance with Dragons, not on the TV show."


def load(name: str) -> dict | None:
    """The saved bundle (model + metadata), or None if that model was not trained yet."""
    path = MODELS_DIR / f"{name}.joblib"
    if not path.exists():
        return None
    try:
        bundle = joblib.load(path)  # only load model files you trust: joblib can run code
    except Exception as error:  # usually a model saved with another scikit-learn version
        warnings.warn(
            f"Could not load {path.name} ({type(error).__name__}: {error}). It was probably "
            f"saved with another scikit-learn version (yours: {sklearn.__version__}): retrain "
            "it on this computer with `python train_got.py`.",
            stacklevel=2,
        )
        return None
    if bundle.get("sklearn_version") != sklearn.__version__:
        warnings.warn(
            f"{path.name} was trained with scikit-learn {bundle.get('sklearn_version')} but "
            f"{sklearn.__version__} is installed: retrain it if predictions look wrong.",
            stacklevel=2,
        )
    return bundle


def to_row(character: dict) -> dict:
    """An API character -> one row of model features ("books": [1, 3] -> book1=1, book3=1...)."""
    row = {key: value for key, value in character.items() if key not in ("name", "books")}
    for book in range(1, 6):
        row[f"book{book}"] = int(book in character.get("books", []))
    for flag in ("male", "noble", "married"):
        row[flag] = int(row[flag])
    if row.get("age") is None:
        row["age"] = float("nan")  # the model fills missing ages itself (and knows they were)
    return row


def comment(probability: float, threshold: float) -> str:
    if probability >= 0.8:
        return "Valar morghulis. All men must die, and you are first in line."
    if probability >= 0.6:
        return "Winter came for you. Whatever you do, decline every wedding invitation."
    if probability >= threshold:
        return "Probably dead by the end of book 5. Trust no one."
    if probability >= threshold / 2:
        return "You survive... for now. Keep your head down and your friends close."
    return "Safe as a maester in the Citadel. Boring, but alive."


def answers(bundle: dict, frame: pd.DataFrame, names: list[str]) -> list[dict]:
    probabilities = bundle["model"].predict_proba(frame[bundle["features"]])[:, 1]
    threshold = bundle["threshold"]
    return [
        {
            "name": name,
            "death_probability": round(float(p), 4),
            "survival_chance": f"{1 - p:.0%}",
            "dies": bool(p >= threshold),
            "verdict": "dies" if p >= threshold else "survives",
            "threshold": threshold,
            "comment": comment(float(p), threshold),
            "warning": WARNING,
        }
        for p, name in zip(probabilities, names, strict=True)
    ]


def predict_got(bundle: dict, characters: list[dict]) -> list[dict]:
    frame = pd.DataFrame([to_row(c) for c in characters], columns=bundle["features"])
    return answers(bundle, frame, [c.get("name", "Unnamed") for c in characters])


def lookup(bundle: dict, search: str, limit: int = 10) -> list[dict]:
    """Real characters whose name contains `search`: what the model says vs. what the books say."""
    known = bundle["characters"]
    found = known[known["name"].str.contains(search, case=False, regex=False)].head(limit)
    if found.empty:
        return []
    results = answers(bundle, found, found["name"].tolist())
    for result, (_, row) in zip(results, found.iterrows(), strict=True):
        actually_dies = bool(row["dies"])
        result["books_say"] = "dead" if actually_dies else "alive"
        result["model_was_right"] = result["dies"] == actually_dies
        result["in_training"] = bool(row["in_training"])
    return results
