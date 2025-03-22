from fastapi import FastAPI
from app.routes import predict

app = FastAPI(title="ML Model API")

app.include_router(predict.router)

@app.get("/")
def home():
    return {"message": "Welcome to the ML API"}
