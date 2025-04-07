from fastapi import APIRouter
from app.models import LoanRequest, LoanResponse, ModelInput
from app.services import MLService
from app.services import OpenAIService
from app.services import LlamaService


router = APIRouter()

@router.post("/ml/predict")
def get_prediction(data: ModelInput):  
    return {"model": data.model_type, "prediction": MLService.predict(data)}


# @router.post("/open-api/predict", response_model=LoanResponse)
@router.post("/open-ai/predict", response_model=None)
async def get_fine_tune_open_ai_prediction(data: LoanRequest):  
    prediction = await OpenAIService.predict(data) 
    print('prediction', prediction)
    return {"model": 'OpenAi Fine tune model', "prediction": prediction}

@router.post("/llama/predict", response_model=None)
async def get_fine_tune_llama_prediction(data: LoanRequest):  
    prediction = await LlamaService.predict(data) 
    print('prediction', prediction)
    return {"model": 'Llama Fine tune model', "prediction": prediction}