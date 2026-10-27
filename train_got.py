"""Train the Game of Thrones survival model: will this character die? (run: python train_got.py)

The same steps as notebooks/1_got_model.ipynb, in one script.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = Path(__file__).parent
DATA = HERE / "data" / "got" / "character-predictions.csv"
MODELS = HERE / "models"

TARGET = "dies"  # 1 = dead by the end of book 5, 0 = still alive (the CSV's isAlive, flipped)

# Columns we must NOT learn from: they give the answer away (leakage) or are just row ids.
LEAKS = {
    "actual": "a copy of isAlive",
    "pred": "the Munich team's prediction (copying it would be cheating)",
    "alive": "the Munich team's probability of being alive",
    "plod": "the Munich team's probability of death",
    "DateoFdeath": "only dead characters have a date of death",
}
# Columns that are (nearly) always empty: 86-99% missing.
EMPTY = ["mother", "father", "heir", "spouse",
         "isAliveMother", "isAliveFather", "isAliveHeir", "isAliveSpouse"]  # fmt: skip
# Columns that repeat another one: isPopular is popularity > 0.5 (98% of rows),
# boolDeadRelations is numDeadRelations > 0 (100%).
REPEATS = ["isPopular", "boolDeadRelations"]

# 347 houses is too many for 1,946 characters: keep the big ones, the rest become "other".
HOUSES = ["night's watch", "frey", "stark", "targaryen", "lannister", "greyjoy", "tyrell",
          "martell", "arryn", "baratheon", "bolton", "tully"]  # fmt: skip
# 64 spellings of culture ("Northmen", "northmen", "Free folk", "Lyseni"...) -> 14 groups.
CULTURES = {
    "northmen": "northmen", "crannogmen": "northmen", "northern mountain clans": "northmen",
    "first men": "northmen",
    "ironborn": "ironborn", "ironmen": "ironborn",
    "free folk": "free folk", "wildling": "free folk", "wildlings": "free folk",
    "rivermen": "rivermen", "riverlands": "rivermen",
    "valemen": "vale", "vale": "vale", "vale mountain clans": "vale", "sistermen": "vale",
    "westerman": "westermen", "westermen": "westermen", "westerlands": "westermen",
    "reach": "reach", "reachmen": "reach", "the reach": "reach",
    "stormlands": "stormlands", "stormlander": "stormlands",
    "dornish": "dornish", "dornishmen": "dornish", "dorne": "dornish", "rhoynar": "dornish",
    "westeros": "andals", "andal": "andals", "andals": "andals",
    "valyrian": "valyrian",
    "braavosi": "free cities", "braavos": "free cities", "lysene": "free cities",
    "lyseni": "free cities", "myrish": "free cities", "pentoshi": "free cities",
    "tyroshi": "free cities", "qohor": "free cities", "norvoshi": "free cities",
    "norvos": "free cities",
    "ghiscari": "ghiscari", "ghiscaricari": "ghiscari", "astapori": "ghiscari",
    "astapor": "ghiscari", "meereenese": "ghiscari", "meereen": "ghiscari",
    "dothraki": "dothraki",
}  # fmt: skip
TITLES = ["none", "knight", "lord or lady", "royalty", "maester", "clergy", "office"]

CATEGORIES = {
    "house": [*HOUSES, "other", "none"],
    "culture": [*sorted(set(CULTURES.values())), "other", "unknown"],
    "title": TITLES,
}
CATEGORICAL = list(CATEGORIES)
NUMERIC = ["male", "noble", "married", "age", "dead_relatives", "popularity",
           "book1", "book2", "book3", "book4", "book5"]  # fmt: skip
FEATURES = CATEGORICAL + NUMERIC
BOOKS = {
    1: "A Game of Thrones", 2: "A Clash of Kings", 3: "A Storm of Swords",
    4: "A Feast for Crows", 5: "A Dance with Dragons",
}  # fmt: skip


def house_group(house: object) -> str:
    if pd.isna(house):
        return "none"
    name = str(house).removeprefix("House ").strip().lower()
    if name.startswith("baratheon"):  # "Baratheon of Dragonstone", "of King's Landing"
        return "baratheon"
    return name if name in HOUSES else "other"


def culture_group(culture: object) -> str:
    if pd.isna(culture):
        return "unknown"
    return CULTURES.get(str(culture).strip().lower(), "other")


OFFICES = ("hand of", "master of", "commander", "ranger", "steward", "seneschal", "castellan",
           "cupbearer", "bloodrider", "captain", "brother", "good master", "goodwife", "magister",
           "wisdom")  # fmt: skip
ROYAL = re.compile(r"queen|prince|khal|emperor|\bking\b|^king|seven kingdoms|^andals$")


def title_group(title: object) -> str:
    if pd.isna(title):
        return "none"
    t = str(title).lower()
    if t in ("ser", "knight") or t.startswith("ser "):
        return "knight"
    if "maester" in t:
        return "maester"
    if "sept" in t or "priest" in t or "devout" in t:
        return "clergy"
    if any(office in t for office in OFFICES):
        return "office"  # a job at court or in the Night's Watch, not a rank
    if ROYAL.search(t):
        return "royalty"
    # Everything else is a lord or lady: the wiki often stores only the seat ("Winterfell" for
    # the Lord of Winterfell, "Casterly Rock" for the Lord of Casterly Rock).
    return "lord or lady"


def load(path: Path = DATA) -> pd.DataFrame:
    """One row per character: `name`, the features with readable names, and the target `dies`."""
    raw = pd.read_csv(path)
    age = raw["age"].where(raw["age"].between(0, 120))  # two broken birth dates give -298,001
    df = pd.DataFrame(
        {
            "name": raw["name"],
            "house": raw["house"].map(house_group),
            "culture": raw["culture"].map(culture_group),
            "title": raw["title"].map(title_group),
            "male": raw["male"],
            "noble": raw["isNoble"],
            "married": raw["isMarried"],
            "age": age,
            "dead_relatives": raw["numDeadRelations"],
            "popularity": raw["popularity"],
            **{f"book{i}": raw[f"book{i}"] for i in BOOKS},
        }
    )
    df[TARGET] = 1 - raw["isAlive"]
    return df


def make_model(estimator, scale: bool = False) -> Pipeline:
    """One-hot the categories, fill the missing ages (and flag them), then the estimator."""
    numeric = [SimpleImputer(strategy="median", add_indicator=True)]
    if scale:
        numeric.append(StandardScaler())
    prepare = ColumnTransformer(
        [
            ("categories", OneHotEncoder(categories=list(CATEGORIES.values())), CATEGORICAL),
            ("numbers", make_pipeline(*numeric), NUMERIC),
        ]
    )
    return Pipeline([("prepare", prepare), ("model", estimator)])


def step(title: str) -> None:
    print(f"\n=== {title} " + "=" * max(0, 60 - len(title)))


def train(data_path: Path = DATA, models_dir: Path = MODELS, verbose: bool = True) -> dict:
    say = print if verbose else (lambda *args, **kwargs: None)
    show = step if verbose else (lambda title: None)

    show("1. Load the data, drop the leaks, clean the rest")
    df = load(data_path)
    say(f"{len(df)} characters, {len(FEATURES)} features")
    say(f"dropped (leak): {', '.join(LEAKS)}")
    say(f"dropped (empty or repeated): {len(EMPTY) + len(REPEATS)} columns")
    say(f"dead: {df[TARGET].mean():.1%}  <- only 1 in 4 dies: the classes are unbalanced")

    show("2. Split: train on 80%, keep 20% aside for the final test")
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    say(f"train: {len(X_train)} rows, test: {len(X_test)} rows")

    show("3. Baselines: a guess, and a one-line rule a fan could use")
    guess = np.zeros(len(y_test), dtype=int)  # "everyone survives"
    say(f"'everyone survives':  accuracy {accuracy_score(y_test, guess):.1%}, "
        f"deaths caught: 0 of {int(y_test.sum())}")  # fmt: skip
    rule = ((X_test["book4"] == 0) & (X_test["book5"] == 0)).astype(int)
    say(f"'not in book 4 or 5 -> dead': accuracy {accuracy_score(y_test, rule):.1%}, "
        f"deaths caught: {int((rule & y_test).sum())} of {int(y_test.sum())}, "
        f"F1 {f1_score(y_test, rule):.2f}")  # fmt: skip
    say("-> accuracy flatters 'everyone survives': judge the models on how well they find deaths")

    show("4. Compare three models with 5-fold cross-validation")
    candidates = {
        "logistic regression": make_model(LogisticRegression(max_iter=2000), scale=True),
        "random forest": make_model(
            RandomForestClassifier(n_estimators=300, min_samples_leaf=3, random_state=42, n_jobs=-1)
        ),
        "gradient boosting": make_model(
            HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, random_state=42)
        ),
    }
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = {}
    for name, model in candidates.items():
        cv = cross_validate(model, X_train, y_train, cv=folds, scoring=["roc_auc", "neg_log_loss"])
        auc, loss = cv["test_roc_auc"], -cv["test_neg_log_loss"]
        scores[name] = auc.mean()
        say(
            f"{name:<20} ROC AUC {auc.mean():.3f} (+/- {auc.std():.3f})  log loss {loss.mean():.3f}"
        )
    best_name = max(scores, key=scores.get)
    model = candidates[best_name]
    say(f"-> best ROC AUC (1.0 = perfect ranking, 0.5 = coin flip): {best_name}")

    show("5. Choose the threshold (on training data, never on the test set)")
    # Probabilities each training row gets from a model that did not see it.
    proba_cv = cross_val_predict(model, X_train, y_train, cv=folds, method="predict_proba")[:, 1]
    # 0.5 calls too few deaths when only 1 in 4 dies: pick the threshold with the best F1 for
    # deaths, the balance between "deaths we catch" (recall) and "death calls that are right"
    # (precision).
    grid = np.round(np.arange(0.10, 0.71, 0.01), 2)
    f1s = [f1_score(y_train, proba_cv >= t) for t in grid]
    threshold = float(grid[int(np.argmax(f1s))])
    say(f"threshold 0.50: F1 {f1_score(y_train, proba_cv >= 0.5):.3f}")
    say(f"threshold {threshold:.2f}: F1 {max(f1s):.3f}  <- chosen")
    say("(called 'dies' when the probability of death is at or above the threshold)")

    show("6. Train on all training data, evaluate once on the test set")
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    predicted = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, predicted).ravel()
    metrics = {
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
        "accuracy": round(accuracy_score(y_test, predicted), 4),
        "precision": round(precision_score(y_test, predicted), 4),
        "recall": round(recall_score(y_test, predicted), 4),
        "f1": round(f1_score(y_test, predicted), 4),
        "baseline_accuracy": round(accuracy_score(y_test, guess), 4),
        "deaths_missed": int(fn),
        "survivors_killed": int(fp),
    }
    for name, value in metrics.items():
        say(f"{name:<18} {value}")
    say(f"deaths caught: {tp} of {tp + fn}; survivors wrongly killed: {fp} of {fp + tn}")

    show("7. Which features matter (permutation importance on the test set)")
    importance = permutation_importance(
        model, X_test, y_test, scoring="roc_auc", n_repeats=10, random_state=42, n_jobs=-1
    )
    ranked = sorted(zip(FEATURES, importance.importances_mean, strict=True), key=lambda x: -x[1])
    for feature, value in ranked[:6]:
        say(f"{feature:<16} {value:+.3f}")

    show("8. Save the model")
    characters = df.assign(in_training=df.index.isin(X_train.index))
    bundle = {
        "model": model,
        "algorithm": best_name,
        "features": FEATURES,
        "categories": CATEGORIES,
        "threshold": threshold,
        "metrics": metrics,
        "importance": {feature: round(float(value), 4) for feature, value in ranked},
        "characters": characters,  # for looking up real characters in the API
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
    }
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / "got.joblib"
    joblib.dump(bundle, path)
    say(f"saved to {path}")
    return bundle


if __name__ == "__main__":
    train()
