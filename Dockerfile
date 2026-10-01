# SL-RAG app image: the API, live /ws/stream pipeline and trace UI on :8000.
# Evaluation data (eval/), tests and run outputs are not copied in (see .dockerignore);
# docker-compose.yml mounts eval/ read-only at run time for the UI's scenario picker.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
COPY pyproject.toml README.md ./
COPY slrag ./slrag
COPY config.yaml ./
COPY data ./data
COPY scripts/fetch_models.py ./scripts/fetch_models.py
# CPU-only torch first (the default wheel pulls CUDA), then the package; the dense retrieval model is
# downloaded at build time into /app/models and loaded offline at run time.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install -e . \
    && python scripts/fetch_models.py \
    && useradd --create-home --uid 1000 slrag \
    && mkdir -p /app/out \
    && chown -R slrag:slrag /app
USER slrag

EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=30s --retries=6 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["python", "-m", "uvicorn", "slrag.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
