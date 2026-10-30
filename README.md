# Game of Thrones survival: build a model, serve it with FastAPI, ship it in Docker

Will this character **survive or die**? A model built with plain **scikit-learn** from what the
books tell us about 1,946 characters of *A Song of Ice and Fire* (house, culture, title, family,
popularity, which books they appear in), saved with **joblib**, served by a small **FastAPI**
app, and packaged in a **Docker** image that anyone can run with one command.

Same structure and stack as project 1 (mushrooms), plus Docker: no MLflow, no database, a few
hundred lines of Python you can read in one sitting.

> **Spoilers**, up to book 5 (*A Dance with Dragons*). The data comes from the books, not the
> TV show.

```
ai-engineering-project-2/
├── data/got/
│   ├── character-predictions.csv     1,946 characters (Kaggle, "Game of Thrones")
│   └── README.md                     where it comes from, every column, what is used
├── notebooks/
│   └── 1_got_model.ipynb             build the model step by step, with plots
├── train_got.py                      the same as the notebook, in one script
├── models/                           got.joblib (created by training)
├── app/
│   ├── ml.py                         load the model, make predictions (no FastAPI here)
│   ├── schemas.py                    what the API accepts and returns (Pydantic)
│   └── main.py                       the FastAPI app: routes, loading the model at startup
├── tests/                            pytest: data cleaning, training quality, every endpoint
├── Dockerfile                        the image: install, train, serve
├── docker-compose.yml                build and start it with one command
├── .dockerignore                     what stays out of the image
└── requests.http                     ready-made requests (VS Code REST Client, JetBrains)
```

## Quick start with Docker (nothing else to install)

Install [Docker Desktop](https://www.docker.com/products/docker-desktop/), start it, then in
this folder:

```bash
docker compose up --build
```

The first build takes a few minutes (it installs the libraries and trains the model). Then open
<http://localhost:8000/docs>. Stop it with `Ctrl+C`, or `docker compose down`.

## Setup without Docker

Python 3.10 or newer.

```bash
cd ai-engineering-project-2
python -m venv .venv
.venv\Scripts\activate            # macOS / Linux: source .venv/bin/activate
pip install -r requirements-dev.txt
```

`requirements.txt` is what the API needs (the Docker image installs only this);
`requirements-dev.txt` adds the notebook, plots, tests and linter.

## Part 1: build the model

```bash
jupyter lab notebooks             # step by step, with explanations and plots
python train_got.py               # ~30 s, prints every step
```

**In Google Colab:** upload `notebooks/1_got_model.ipynb`, run all cells, and upload
`character-predictions.csv` when it asks. The last cells save `got.joblib` and download it: put
it in `models/`. **Colab must use the same scikit-learn version as your computer** (`pip show scikit-learn`),
otherwise the saved model may not load: run `!pip install scikit-learn==<your version>` in Colab
first, restart the session, then run all. When in doubt, run `python train_got.py` locally.

Both write `models/got.joblib`. It holds the fitted model **and** what the API needs to use
it: the feature order, the allowed categories, the threshold, the scores, the training date,
the scikit-learn version and the characters (for looking them up).

| Step | What it teaches |
|---|---|
| 1. Load, drop leaks, clean | **Leakage:** the file contains the Munich team's own predictions (`pred`, `plod`...), a copy of the answer (`actual`) and a date of death that only the dead have. Train on them and you score ~100% for nothing. Also: 99%-empty family columns, two ages of −298,001, 64 spellings of 14 cultures, 347 houses. |
| 2. Train / test split | Judge a model on characters it has never seen; `stratify` keeps 25% dead in both parts. |
| 3. Baselines | "Everyone survives" scores 74.6% accuracy and finds **zero** deaths. "Not in book 4 or 5 → dead" has F1 0.48. **Accuracy lies when the classes are unbalanced.** |
| 4. Cross-validation | Logistic regression, random forest, gradient boosting inside a `Pipeline` (one-hot encoding, fill missing ages). Random forest wins: ROC AUC **0.825**. |
| 5. Threshold | Only 1 in 4 dies, so the default 0.5 misses most deaths. The threshold with the best F1 on cross-validated training predictions: **0.31** (F1 0.61 instead of 0.47). |
| 6. Test set, once | ROC AUC **0.83**, accuracy 76.7%; catches 71 of 99 deaths, "kills" 63 of 291 survivors. |
| 7. What matters | Appearing in **book 4** (you made it through books 1-3), **popularity**, age, house. |
| 8. Save | One `.joblib` bundle with everything the API needs. |

## Part 2: serve it with FastAPI

```bash
fastapi dev app/main.py           # http://127.0.0.1:8000, reloads when the code changes
```

Open <http://127.0.0.1:8000/docs>: every endpoint with an example you can send from the page
("Try it out").

| Method | Path | Does |
|---|---|---|
| GET | `/health` | Is the API up, is the model loaded |
| GET | `/models` | How the model was trained, its scores, which features matter |
| POST | `/got/predict` | One character (real or invented) → probability of death, verdict, a comment |
| POST | `/got/predict/batch` | A list of characters in one call |
| GET | `/got/characters?search=Stark` | Real characters: what the model says vs. what the books say |

Two real characters from the data, with their values exactly as the dataset records them.

**Arya Stark: survives** (a test-set character: the model never saw her during training)

```bash
curl -s http://127.0.0.1:8000/got/predict -H "Content-Type: application/json" -d '{
  "name": "Arya Stark", "male": false, "house": "stark", "culture": "northmen",
  "title": "royalty", "noble": true, "married": false, "age": 16, "dead_relatives": 8,
  "popularity": 1.0, "books": [1, 2, 3, 4, 5]
}'
# {"name":"Arya Stark","death_probability":0.2256,"survival_chance":"77%",
#  "dies":false,"verdict":"survives","threshold":0.31,
#  "comment":"You survive... for now. Keep your head down and your friends close.", ...}
```

**Eddard (Ned) Stark: dies** (a training-set character; the books agree)

```bash
curl -s http://127.0.0.1:8000/got/predict -H "Content-Type: application/json" -d '{
  "name": "Eddard Stark", "male": true, "house": "stark", "culture": "northmen",
  "title": "lord or lady", "noble": true, "married": true, "age": 36, "dead_relatives": 6,
  "popularity": 1.0, "books": [1, 2, 3, 4, 5]
}'
# {"name":"Eddard Stark","death_probability":0.4272,"survival_chance":"57%",
#  "dies":true,"verdict":"dies","threshold":0.31,
#  "comment":"Probably dead by the end of book 5. Trust no one.", ...}
```

Ned is listed in all five books although he dies in book 1: the wiki counts every book that
mentions a character. Probabilities can differ slightly with another scikit-learn version.

Every field has a default, so you can leave out what you don't know (`{}` is a valid character).
More requests, including invalid ones, are in [`requests.http`](requests.http).

> **Try it in class:** everyone invents themselves as a character. Who survives? Then look up
> `?search=Jon Snow` and `?search=Robb Stark`. Note `in_training`: the model has already
> seen most real characters, so being right about them proves little. Only the test-set score
> counts.

### How the app is built

1. [`app/ml.py`](app/ml.py): plain Python, no web code. `load()` reads the `.joblib` file,
   `to_row()` turns an API character into model features (`"books": [1, 3]` → `book1=1,
   book3=1`...), `predict_got()` and `lookup()` return the answers.
2. [`app/schemas.py`](app/schemas.py): Pydantic models. `house`, `culture` and `title` are
   `Literal`s of the values the model knows, numbers have ranges (`age` 0-120, `popularity`
   0-1), books are 1-5. Bad input → `422` before the model ever sees it. A test checks these
   values match the training script's `CATEGORIES`.
3. [`app/main.py`](app/main.py): thin routes. The model is **loaded once at startup**
   (`lifespan`); a model that was not trained answers `503` with the command to run.

## Part 3: test it

```bash
pytest
ruff check .
```

The tests train the model into a temporary folder once (they never touch `models/`), check the
cleaning (no leaks, no broken ages, every category known), check the model beats the baselines,
and call every endpoint: predictions, a batch, defaults, real characters, invalid input (`422`)
and a missing model (`503`).

## Part 4: ship it with Docker

A container packs the code, the libraries, Python itself **and the trained model** into one
image that runs the same on any computer: yours, a classmate's, or a cloud server.

```bash
docker compose up --build         # build the image and start the API on port 8000
docker compose up -d              # ...or in the background
docker compose logs -f            # watch the logs
docker compose down               # stop and remove the container
```

Without Compose, the same in two commands:

```bash
docker build -t got-survival-api .
docker run --rm -p 8000:8000 got-survival-api
```

How the [`Dockerfile`](Dockerfile) works, line by line:

| Step | Why |
|---|---|
| `FROM python:3.12-slim` | A small official image with Python. |
| `COPY requirements.txt` + `pip install` **first** | Docker caches each step. Code changes don't reinstall the libraries; only a new `requirements.txt` does. |
| `COPY data/ train_got.py` + `RUN python train_got.py` | The model is **trained during the build**, with the exact scikit-learn that will load it. No version mismatch, and no `.joblib` file from outside to trust. |
| `COPY app/` **last** | Changing the API doesn't retrain the model. |
| `USER appuser` | Never run a web server as root. |
| `HEALTHCHECK` | Docker calls `/health` every 30 s; `docker ps` shows `healthy` or `unhealthy`. |
| `CMD fastapi run ... --workers 2` | Production mode: no auto-reload, two worker processes. |

[`.dockerignore`](.dockerignore) keeps the virtual environment, git history, notebooks, tests
and old model files out of the image, so it stays small and builds fast.

Useful checks:

```bash
docker ps                         # is it running? is it "healthy"?
docker images got-survival-api    # how big is the image?
docker exec -it got-survival-api python -c "import sklearn; print(sklearn.__version__)"
```

## Exercises

On the model (more at the end of the notebook):

1. Train without `book1` ... `book5`. How much worse is it? Why are they so strong?
2. Put the leaky `plod` column back. Watch the score jump, and explain why it means nothing.
3. Choose the threshold to catch 90% of deaths. How many survivors get "killed"?

On the API:

4. Add `"reasons"` to the answer: the 3 features that pushed this character towards death.
5. Add `GET /got/houses`: the death rate of every house, from the data.
6. Require an `X-API-Key` header (a FastAPI dependency) on the predict endpoints.

On Docker:

7. Make the image smaller with a **multi-stage build**: train in one stage, copy only the
   `.joblib` and the app into the final one. Compare `docker images` before and after.
8. Pass the number of workers as an environment variable in `docker-compose.yml`.
9. Push the image to Docker Hub or Amazon ECR and run it on a cloud server.

## Good to know

- **Leakage is the biggest trap here.** Always ask of every column: *would I know this before
  the answer?*
- **Only load model files you trust.** joblib (like pickle) can run code when it loads a file.
  Training inside the Docker build avoids copying model files around.
- **Train and serve with the same scikit-learn version.** The bundle records the version and the
  API warns when it differs. Docker solves this for good: both happen in the same image.
- **The data has quirks.** Many long-dead historical characters (old Starks, ancient kings) are
  marked "alive", and big characters such as Robb Stark have no house. The model learns these
  quirks too.
- **The books are not finished.** "Alive" means alive at the end of book 5.
- `models/*.joblib` are not committed: anyone can recreate them with the training script.
