# Simple Dockerfile to run tests and the Streamlit app
# Usage examples (optional):
#   docker build -t fxlt .
#   docker run --rm -it -p 8501:8501 --env-file .env fxlt
#   docker run --rm -it fxlt python -m pytest -q -k "fast or safety or filters or pipeline or observability"

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps (minimal). Add more if some wheels require build tools.
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY . .

# Health check target (fast tests)
RUN python -m pytest -q -k "fast or safety or filters or pipeline or observability" || true

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
