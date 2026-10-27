"""What the API accepts and returns. FastAPI validates requests against these and documents
them at /docs."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# The values the model was trained on (`CATEGORIES` in train_got.py; a test checks they match).
House = Literal[
    "night's watch", "frey", "stark", "targaryen", "lannister", "greyjoy", "tyrell", "martell",
    "arryn", "baratheon", "bolton", "tully", "other", "none",
]  # fmt: skip
Culture = Literal[
    "andals", "dornish", "dothraki", "free cities", "free folk", "ghiscari", "ironborn",
    "northmen", "reach", "rivermen", "stormlands", "vale", "valyrian", "westermen", "other",
    "unknown",
]  # fmt: skip
Title = Literal["none", "knight", "lord or lady", "royalty", "maester", "clergy", "office"]
Book = Literal[1, 2, 3, 4, 5]

EXAMPLE_CHARACTER = {
    "name": "Ser Sovann of Phnom Penh",
    "male": True,
    "house": "stark",
    "culture": "northmen",
    "title": "knight",
    "noble": True,
    "married": False,
    "age": 21,
    "dead_relatives": 3,
    "popularity": 0.6,
    "books": [1, 2, 3],
}


class Character(BaseModel):
    """A character, real or invented. Everything has a default: leave out what you don't know."""

    model_config = {"extra": "forbid", "json_schema_extra": {"examples": [EXAMPLE_CHARACTER]}}

    name: str = Field("Unnamed", max_length=80, description="Only used in the answer")
    male: bool = True
    house: House = Field("none", description='"none" = no house, "other" = a smaller house')
    culture: Culture = "unknown"
    title: Title = Field("none", description='"office" = a job like Hand of the King or steward')
    noble: bool = False
    married: bool = False
    age: int | None = Field(None, ge=0, le=120, description="Leave out if unknown")
    dead_relatives: int = Field(0, ge=0, le=20, description="How many relatives already died")
    popularity: float = Field(0.0, ge=0, le=1, description="0 = nobody, 1 = Jon Snow")
    books: list[Book] = Field(
        [], description="Books the character appears in (1 = A Game of Thrones ... 5 = A Dance "
        "with Dragons)",
    )  # fmt: skip


class Prediction(BaseModel):
    name: str
    death_probability: float = Field(description="Probability of dying by the end of book 5")
    survival_chance: str = Field(description="1 - death_probability, as a percentage")
    dies: bool = Field(description="death_probability >= threshold")
    verdict: Literal["dies", "survives"]
    threshold: float = Field(description="The decision threshold chosen during training")
    comment: str
    warning: str


class RealCharacter(Prediction):
    books_say: Literal["dead", "alive"] = Field(description="What actually happened (book 5)")
    model_was_right: bool
    in_training: bool = Field(
        description="The model learned from this character: being right is no surprise"
    )


class ModelInfo(BaseModel):
    name: str
    loaded: bool
    algorithm: str | None = None
    trained_at: str | None = None
    threshold: float | None = None
    metrics: dict[str, float] = {}
    importance: dict[str, float] = {}
