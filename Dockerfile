# syntax=docker/dockerfile:1
# Multi-stage build for the Adaptive Offers decision service.
# Course: Containers (Dockerfile, images & registry, networks).

# ---- builder: install deps into a wheel cache ------------------------------
FROM python:3.11-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
# Build a wheel so the runtime image stays slim and reproducible.
RUN pip install --upgrade pip build && pip wheel --no-deps -w /wheels .

# ---- runtime: minimal image -----------------------------------------------
FROM python:3.11-slim AS runtime
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    APP_ENV=prod \
    API_HOST=0.0.0.0 \
    API_PORT=8000 \
    MLFLOW_TRACKING_URI=file:/app/mlruns

# Non-root user for security.
RUN useradd --create-home --uid 10001 appuser
WORKDIR /app

COPY --from=builder /wheels /wheels
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY data/golden_set ./data/golden_set
# Install the runtime deps + the prebuilt wheel.
RUN pip install --no-cache-dir "adaptive-offers @ /wheels/$(ls /wheels | grep adaptive_offers)" \
    && pip install --no-cache-dir "uvicorn[standard]" \
    && rm -rf /wheels

USER appuser
EXPOSE 8000

# Container healthcheck hits the readiness endpoint.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

# Default: serve the API. Override CMD to run the CLI pipeline if desired.
CMD ["uvicorn", "adaptive_offers.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
