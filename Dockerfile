# The Game of Thrones Survival API in a container.
#   docker build -t got-survival-api .
#   docker run --rm -p 8000:8000 got-survival-api      -> http://localhost:8000/docs
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# 1. Dependencies first: Docker caches this layer until requirements.txt changes,
#    so editing the code does not reinstall everything.
COPY requirements.txt .
RUN pip install -r requirements.txt

# 2. Train the model inside the image: it is built with the exact scikit-learn that serves it,
#    and no .joblib file has to be copied in (or trusted) from outside.
COPY data/ data/
COPY train_got.py .
RUN python train_got.py

# 3. The app last: it changes most often, and changing it does not retrain the model.
COPY app/ app/

# Never run a web server as root.
RUN useradd --create-home --uid 1000 appuser
USER appuser

EXPOSE 8000

# Docker marks the container "unhealthy" if /health stops answering.
HEALTHCHECK --interval=30s --timeout=3s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["fastapi", "run", "app/main.py", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
