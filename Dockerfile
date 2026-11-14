# Serving image for the plasma-etch VM API (CPU only, no torch).
#   pevm export                                   # produces outputs/model_bundle.joblib
#   docker build -t plasma-etch-vm:0.1 .
#   docker run --rm -p 8000:8000 plasma-etch-vm:0.1
#   curl localhost:8000/health
# Expected image size ~0.8-1.0 GB (python:3.12-slim + numpy/pandas/sklearn/lightgbm/pyarrow/fastapi).
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PEVM_BUNDLE=/app/outputs/model_bundle.joblib \
    PEVM_FEATURE_STORE=/app/data/features/wafer_features.parquet

# libgomp is needed by LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
RUN pip install ".[serve]"

# model artefacts (build context must contain them: run `pevm export` first)
COPY outputs/model_bundle.joblib outputs/model_card.json ./outputs/
# optional: feature store for GET /predict/{wafer_id}
COPY data/features/wafer_features.parquet data/features/wafer_targets.parquet ./data/features/

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1
CMD ["uvicorn", "plasma_etch_vm.serve.api:app", "--host", "0.0.0.0", "--port", "8000"]
