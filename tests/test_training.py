"""The training script cleans the data, never learns from the leaks, beats the baselines, and
the API accepts exactly the values the model was trained on."""

import typing

import joblib

import train_got
from app.schemas import Character


def test_got_model(models_dir):
    bundle = joblib.load(models_dir / "got.joblib")
    metrics = bundle["metrics"]
    assert metrics["roc_auc"] > 0.75  # a coin flip is 0.5
    assert metrics["recall"] > 0.5  # it finds most deaths ("everyone survives" finds none)
    assert metrics["f1"] > 0.5
    assert 0.1 <= bundle["threshold"] < 0.5  # few deaths in the data: below 0.5
    assert bundle["features"] == train_got.FEATURES
    assert len(bundle["characters"]) == 1946


def test_no_leaks_in_features():
    raw_names = {"actual", "pred", "alive", "plod", "DateoFdeath", "isAlive", "S.No"}
    assert not raw_names & set(train_got.FEATURES)
    assert "dies" not in train_got.FEATURES


def test_data_is_clean():
    df = train_got.load()
    assert len(df) == 1946
    assert set(df["dies"]) == {0, 1}
    assert df["age"].dropna().between(0, 120).all()  # the -298,001 ages are gone
    assert not df.drop(columns="age").isna().any().any()  # only age may be missing
    for feature, allowed in train_got.CATEGORIES.items():
        assert set(df[feature]) <= set(allowed), feature


def test_groups():
    assert train_got.house_group("House Baratheon of Dragonstone") == "baratheon"
    assert train_got.house_group("Night's Watch") == "night's watch"
    assert train_got.house_group("House Osgrey") == "other"
    assert train_got.culture_group("northmen") == train_got.culture_group("Northmen")
    assert train_got.title_group("Ser") == "knight"
    assert train_got.title_group("Grand Maester") == "maester"
    assert train_got.title_group("King in the North") == "royalty"
    assert train_got.title_group("Hand of the King") == "office"
    assert train_got.title_group("Lord of Kingsgrave") == "lord or lady"
    assert train_got.title_group("Winterfell") == "lord or lady"  # only the seat is stored


def test_schema_matches_training_categories():
    fields = Character.model_fields
    for feature, allowed in train_got.CATEGORIES.items():
        assert set(typing.get_args(fields[feature].annotation)) == set(allowed), feature
    numeric = {f for f in train_got.NUMERIC if not f.startswith("book")}
    assert numeric <= set(fields)
    assert "books" in fields
