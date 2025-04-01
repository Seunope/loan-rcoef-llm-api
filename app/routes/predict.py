from fastapi import APIRouter
from app.models import LoanRequest, LoanResponse, ModelInput
from app.services.model_service import predict
from app.services.open_ai_service import predict_repayment

router = APIRouter()

@router.post("/ml/predict")
def get_prediction(data: ModelInput):  
    return {"model": data.model_type, "prediction": predict(data)}


# @router.post("/open-api/predict", response_model=LoanResponse)
@router.post("/open-ai/predict", response_model=None)
async def get_fine_tune_open_ai_prediction(data: LoanRequest):  
    prediction = await predict_repayment(data) 
    print('prediction', prediction)
    return {"model": 'OpenAi Fine tune model', "prediction": prediction}
