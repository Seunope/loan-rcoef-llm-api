from fastapi import APIRouter
from app.models import ModelInput
from app.services.model_service import predict

router = APIRouter()

@router.post("/predict/")
def get_prediction(model_type: str, data: ModelInput):
    return {"model": model_type, "prediction": predict(model_type, data.features)}
