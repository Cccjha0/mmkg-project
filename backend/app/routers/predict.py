from __future__ import annotations

from fastapi import APIRouter

from app.schemas.predict import TailPredictionRequest, TailPredictionResponse
from app.services.predict_service import predict_tail

router = APIRouter(prefix="/api/predict", tags=["predict"])


@router.post("/tail", response_model=TailPredictionResponse, summary="Tail prediction")
def api_predict_tail(request: TailPredictionRequest) -> TailPredictionResponse:
    return predict_tail(request)
