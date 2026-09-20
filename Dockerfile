# Base image
FROM python:3.11-slim

# Python settings

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Working directory

WORKDIR /app

# System dependencies

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Application files

COPY . .

# Python dependencies

RUN pip install --no-cache-dir \
    --upgrade pip \
    && pip install --no-cache-dir \
        -r requirements.txt


# FAISS persistent-data directory

RUN mkdir -p /app/data/faiss


# Application port

EXPOSE 8000


# Start FastAPI

CMD [
    "uvicorn",
    "main:app",
    "--host",
    "0.0.0.0",
    "--port",
    "8000"
]