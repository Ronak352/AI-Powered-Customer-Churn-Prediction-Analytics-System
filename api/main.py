"""
api/main.py
===========
Optional REST API for the Customer Churn Prediction system, built with
FastAPI. Reuses the exact same ``src.prediction`` engine as the Streamlit
dashboard, so predictions are guaranteed identical between both interfaces.

Run locally:
    uvicorn api.main:app --reload --port 8000

Interactive docs (Swagger UI) will then be available at:
    http://127.0.0.1:8000/docs

Endpoints:
    POST /predict         - score a single customer
    POST /batch_predict    - score a list of customers
    GET  /metrics          - return the saved model comparison table
    GET  /health           - liveness/readiness probe
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

sys.path.append(str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src import config
from src.evaluation import load_model_comparison
from src.prediction import available_models, predict_batch, predict_single

app = FastAPI(
    title="Customer Churn Prediction API",
    description="AI-Powered Customer Churn Prediction & Analytics System — REST interface.",
    version="1.0.0",
)


class CustomerRecord(BaseModel):
    """Schema for a single raw customer record."""

    CreditScore: int = Field(..., ge=300, le=900, example=650)
    Geography: str = Field(..., example="France")
    Gender: str = Field(..., example="Female")
    Age: int = Field(..., ge=18, le=100, example=40)
    Tenure: int = Field(..., ge=0, le=10, example=5)
    Balance: float = Field(..., ge=0, example=50000.0)
    NumOfProducts: int = Field(..., ge=1, le=4, example=2)
    HasCrCard: int = Field(..., ge=0, le=1, example=1)
    IsActiveMember: int = Field(..., ge=0, le=1, example=1)
    EstimatedSalary: float = Field(..., ge=0, example=100000.0)
    model_name: Optional[str] = Field(None, description="Optional model override")


class BatchRequest(BaseModel):
    records: List[CustomerRecord]
    model_name: Optional[str] = None


class PredictionResponse(BaseModel):
    Prediction: str
    Probability: float
    Risk_Level: str
    Confidence: float
    Model_Used: str
    Recommendations: List[str]


@app.get("/health", tags=["System"])
def health():
    """Liveness/readiness probe."""
    return {
        "status": "ok",
        "dataset_available": config.DATASET_PATH.exists(),
        "models_available": available_models(),
    }


@app.get("/metrics", tags=["System"])
def metrics():
    """Return the saved model comparison table (accuracy, precision, recall, F1, ROC AUC, etc.)."""
    df = load_model_comparison()
    if df.empty:
        raise HTTPException(status_code=404, detail="No trained model metrics found. Run `python -m src.training` first.")
    return df.to_dict(orient="records")


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(customer: CustomerRecord):
    """Score a single customer and return probability, risk level, and recommendations."""
    payload = customer.dict(exclude={"model_name"})
    try:
        result = predict_single(payload, model_name=customer.model_name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return PredictionResponse(
        Prediction="Churn" if result.will_churn else "No Churn",
        Probability=round(result.probability, 4),
        Risk_Level=result.risk_level,
        Confidence=round(result.confidence, 4),
        Model_Used=result.model_used,
        Recommendations=result.recommendations,
    )


@app.post("/batch_predict", tags=["Prediction"])
def batch_predict(request: BatchRequest):
    """Score a batch of customers at once. Returns a list of result records."""
    import pandas as pd

    if not request.records:
        raise HTTPException(status_code=400, detail="No records provided.")

    rows = [r.dict(exclude={"model_name"}) for r in request.records]
    df = pd.DataFrame(rows)

    try:
        results_df = predict_batch(df, model_name=request.model_name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return results_df.to_dict(orient="records")
