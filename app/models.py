from pydantic import BaseModel, Field
from typing import Dict, Any

class ModelInput(BaseModel):
    model_type: str
    features: Dict[str, Any] = Field(..., description="Dictionary containing feature values") 

class LoanRequest(BaseModel):
    maritalStatus: str = Field(..., description="Marital status of the applicant")
    gender: str = Field(..., description="Gender of the applicant")
    state: str = Field(..., description="Location of the applicant (city)")
    age: int = Field(..., description="Age of the applicant")
    loanAmount: int = Field(..., description="Requested loan amount")
    tenorInDays: int = Field(..., description="Loan tenure in days")

class LoanResponse(BaseModel):
    message: str = Field(..., description="A summary")
    meta: str = Field(..., description="Training performance of model")
    riskLevel: str = Field(..., description="low, medium and high")
    recommendation: str = Field(..., description="Model recommendation")
    repaymentProbabilityScore: int = Field(..., description="Prediction of loan repayment likelihood (0-100)")