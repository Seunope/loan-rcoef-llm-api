from pydantic import BaseModel, Field
from typing import Dict, Any

# class ModelInput(BaseModel):
#     features: list[float]

class ModelInput(BaseModel):
    model_type: str
    features: Dict[str, Any] = Field(..., description="Dictionary containing feature values") 