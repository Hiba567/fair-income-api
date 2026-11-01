"""FastAPI service: predictions, fairness report, health and basic monitoring."""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request

from app.schemas import Applicant, Prediction

MODEL_DIR = Path(os.getenv("MODEL_DIR", Path(__file__).resolve().parent.parent / "models"))
THRESHOLD = float(os.getenv("THRESHOLD", "0.5"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("fair-income-api")

state: dict = {}
stats = {"requests": 0, "predictions": 0, "positive_predictions": 0, "total_latency_ms": 0.0}


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = MODEL_DIR / "model.joblib"
    if not model_path.exists():
        raise RuntimeError(f"Model not found at {model_path}. Run: python -m training.train")
    state["model"] = joblib.load(model_path)
    state["report"] = json.loads((MODEL_DIR / "fairness_report.json").read_text())
    state["version"] = state["report"].get("trained_at", "unknown")
    log.info("Model loaded (trained at %s)", state["version"])
    yield
    state.clear()


app = FastAPI(
    title="Fair Income API",
    description="Income classifier served with a built-in fairness audit across sex and race.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def track_latency(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    stats["requests"] += 1
    stats["total_latency_ms"] += (time.perf_counter() - start) * 1000
    return response


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state}


@app.post("/predict", response_model=Prediction)
def predict(applicant: Applicant):
    try:
        row = pd.DataFrame([applicant.model_dump()])
        proba = float(state["model"].predict_proba(row)[0, 1])
    except Exception as exc:  # never return a half-computed answer
        log.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed") from exc

    label = int(proba >= THRESHOLD)
    stats["predictions"] += 1
    stats["positive_predictions"] += label
    return Prediction(probability=round(proba, 4), label=label,
                      threshold=THRESHOLD, model_version=state["version"])


@app.get("/fairness")
def fairness():
    """Fairness audit computed on the held-out test set at training time."""
    return state["report"]


@app.get("/metrics")
def metrics():
    """Simple monitoring: traffic, latency and live positive-prediction rate (drift signal)."""
    n_pred = stats["predictions"]
    return {
        "requests": stats["requests"],
        "predictions": n_pred,
        "positive_rate": (stats["positive_predictions"] / n_pred) if n_pred else None,
        "mean_latency_ms": (stats["total_latency_ms"] / stats["requests"])
        if stats["requests"] else None,
    }
