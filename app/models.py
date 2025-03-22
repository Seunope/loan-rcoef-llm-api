from pydantic import BaseModel

class ModelInput(BaseModel):
    features: list[float]
