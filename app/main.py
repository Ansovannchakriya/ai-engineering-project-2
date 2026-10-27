"""The API that serves the Game of Thrones survival model (run: fastapi dev app/main.py)."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query

from app import ml
from app.schemas import Character, ModelInfo, Prediction, RealCharacter

MODELS: dict[str, dict | None] = {}
HOW_TO_TRAIN = {"got": "python train_got.py"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    for name in HOW_TO_TRAIN:
        MODELS[name] = ml.load(name)
    yield
    MODELS.clear()


app = FastAPI(
    title="Game of Thrones Survival API",
    description="Will this character survive A Song of Ice and Fire? Invent one, or look up a "
    "real one. (Spoilers up to book 5.)",
    version="1.0.0",
    lifespan=lifespan,
)


def model(name: str) -> dict:
    """The loaded model, or a clear error if it has not been trained yet."""
    bundle = MODELS.get(name)
    if bundle is None:
        hint = (
            f"The {name} model is not trained yet, or its file could not be loaded (saved with "
            f"another scikit-learn version?): run `{HOW_TO_TRAIN[name]}`."
        )
        raise HTTPException(status_code=503, detail=hint)
    return bundle


@app.get("/", include_in_schema=False)
def home() -> dict:
    return {"message": "Game of Thrones Survival API", "docs": "/docs", "health": "/health"}


@app.get("/health", tags=["service"])
def health() -> dict:
    return {"status": "ok", "models": {name: bundle is not None for name, bundle in MODELS.items()}}


@app.get("/models", response_model=list[ModelInfo], tags=["service"])
def models() -> list[dict]:
    """Which models are loaded, how they were trained and how well they scored."""
    return [
        {"name": name, "loaded": False}
        if bundle is None
        else {
            "name": name,
            "loaded": True,
            "algorithm": bundle["algorithm"],
            "trained_at": bundle["trained_at"],
            "threshold": bundle["threshold"],
            "metrics": bundle["metrics"],
            "importance": bundle["importance"],
        }
        for name, bundle in MODELS.items()
    ]


@app.post("/got/predict", response_model=Prediction, tags=["game of thrones"])
def predict(character: Character) -> dict:
    """Will this character survive?"""
    return ml.predict_got(model("got"), [character.model_dump()])[0]


@app.post("/got/predict/batch", response_model=list[Prediction], tags=["game of thrones"])
def predict_batch(characters: list[Character]) -> list[dict]:
    """Several characters in one call (one model call for the whole list)."""
    if not characters:
        return []
    return ml.predict_got(model("got"), [c.model_dump() for c in characters])


@app.get("/got/characters", response_model=list[RealCharacter], tags=["game of thrones"])
def characters(
    search: str = Query(min_length=2, max_length=80, examples=["Stark"]),
    limit: int = Query(10, ge=1, le=50),
) -> list[dict]:
    """Real characters whose name contains `search`: the model's guess vs. the books."""
    return ml.lookup(model("got"), search, limit)
