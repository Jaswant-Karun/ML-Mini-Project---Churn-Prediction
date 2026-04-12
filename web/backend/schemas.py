from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    features: Dict[str, Any] = Field(..., description="Feature dictionary for a single customer")


class PredictionResponse(BaseModel):
    prediction: int
    label: str
    churn_probability: Optional[float] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    supabase_enabled: bool
