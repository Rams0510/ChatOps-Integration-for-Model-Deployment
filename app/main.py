"""
Phase 1 — Step 2: FastAPI app wrapping the ML model.
"""
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from contextlib import asynccontextmanager
from typing import Optional
import os

# ── Model loading ────────────────────────────────────────────────────────────

MODEL_PATH = os.getenv("MODEL_PATH", "model.pkl")

ml_model = {}  # shared state loaded once at startup

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, release on shutdown."""
    print(f"Loading model from {MODEL_PATH}...")
    ml_model["classifier"] = joblib.load(MODEL_PATH)
    print("Model loaded successfully.")
    yield
    ml_model.clear()


# ── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="ML Model API",
    description="Phase 1 — Iris classifier served via FastAPI in Docker",
    version="1.0.0",
    lifespan=lifespan,
)

CLASS_NAMES = {0: "setosa", 1: "versicolor", 2: "virginica"}


# ── Schemas ──────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    """
    Features for one Iris sample.
    data: [sepal_length, sepal_width, petal_length, petal_width]
    Example: {"data": [5.1, 3.5, 1.4, 0.2]}
    """
    data: list[float]

    @field_validator("data")
    @classmethod
    def check_length(cls, v):
        if len(v) != 4:
            raise ValueError(f"Expected 4 features, got {len(v)}")
        return v


class PredictResponse(BaseModel):
    prediction: int
    class_name: str
    probabilities: Optional[dict[str, float]] = None


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", tags=["ops"])
def health():
    """Liveness probe — used by Docker and cloud platforms."""
    model_loaded = "classifier" in ml_model
    return {
        "status": "ok" if model_loaded else "degraded",
        "model_loaded": model_loaded,
    }


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
def predict(body: PredictRequest):
    """
    Predict the Iris species from 4 features.
    Returns the class index, class name, and per-class probabilities.
    """
    if "classifier" not in ml_model:
        raise HTTPException(status_code=503, detail="Model not loaded")

    model = ml_model["classifier"]
    arr = np.array(body.data).reshape(1, -1)

    pred_class = int(model.predict(arr)[0])
    proba = model.predict_proba(arr)[0]

    return PredictResponse(
        prediction=pred_class,
        class_name=CLASS_NAMES[pred_class],
        probabilities={CLASS_NAMES[i]: round(float(p), 4) for i, p in enumerate(proba)},
    )
