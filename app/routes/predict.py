from fastapi import APIRouter
from app.models import ModelInput
from app.services.model_service import predict

router = APIRouter()

@router.post("/ml/predict")
def get_prediction(data: ModelInput):  
    return {"model": data.model_type, "prediction": predict(data)}
