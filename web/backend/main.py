from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.preprocess import load_and_clean_data, split_features_target
from web.backend.database import SupabasePredictionStore
from web.backend.schemas import HealthResponse, PredictionRequest, PredictionResponse

ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT_DIR / "model" / "churn_model.pkl"
DATA_CANDIDATE_PATHS = [
    ROOT_DIR / "data" / "Telco-Customer-Churn.csv",
    ROOT_DIR / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv",
]
FRONTEND_DIR = ROOT_DIR / "web" / "frontend"

app = FastAPI(title="Customer Churn API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

store = SupabasePredictionStore()
model = None
feature_columns = []
feature_schema: Dict[str, Dict[str, Any]] = {}


def resolve_data_path() -> Path | None:
    for path in DATA_CANDIDATE_PATHS:
        if path.exists():
            return path
    return None


@app.on_event("startup")
def startup_event() -> None:
    global model, feature_columns, feature_schema

    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)

    data_path = resolve_data_path()
    if data_path:
        df = load_and_clean_data(str(data_path))
        x, _ = split_features_target(df)
        feature_columns = x.columns.tolist()

        schema = {}
        for col in feature_columns:
            if pd.api.types.is_numeric_dtype(x[col]):
                schema[col] = {
                    "type": "number",
                    "example": float(x[col].median()) if x[col].notna().any() else 0.0,
                }
            else:
                values = sorted(x[col].dropna().astype(str).unique().tolist())
                schema[col] = {
                    "type": "categorical",
                    "options": values,
                    "example": values[0] if values else "",
                }
        feature_schema = schema


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=model is not None,
        supabase_enabled=store.enabled,
    )


@app.get("/api/feature-schema")
def get_feature_schema() -> Dict[str, Any]:
    if not feature_schema:
        raise HTTPException(status_code=500, detail="Feature schema unavailable. Check dataset.")
    return {"features": feature_schema}


@app.post("/api/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    if model is None:
        raise HTTPException(status_code=500, detail="Model not found. Train model first.")

    if not feature_columns:
        raise HTTPException(status_code=500, detail="Feature columns unavailable.")

    incoming = request.features

    row = {}
    for col in feature_columns:
        value = incoming.get(col)

        if value is None:
            schema_item = feature_schema.get(col, {})
            value = schema_item.get("example", 0)

        if col == "TotalCharges":
            try:
                value = float(value)
            except (TypeError, ValueError):
                value = 0.0

        row[col] = value

    input_df = pd.DataFrame([row])

    try:
        pred = int(model.predict(input_df)[0])
        prob = None
        if hasattr(model, "predict_proba"):
            prob = float(model.predict_proba(input_df)[0][1])
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc

    label = "Churn" if pred == 1 else "Stay"

    store.log_prediction(
        {
            "prediction": pred,
            "label": label,
            "churn_probability": prob,
            "features": row,
        }
    )

    return PredictionResponse(prediction=pred, label=label, churn_probability=prob)


@app.get("/api/history")
def history(limit: int = Query(default=20, ge=1, le=200)) -> Dict[str, Any]:
    data = store.get_history(limit=limit)
    return {"rows": data, "supabase_enabled": store.enabled}


app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIR)), name="assets")


@app.get("/")
def serve_home() -> FileResponse:
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return FileResponse(index_path)
