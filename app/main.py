from fastapi import FastAPI
from app.routes import predict
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="ML Model API")

origins = [
    "http://localhost:3000",  # If frontend is running on React/Vue
    "http://127.0.0.1:8001",  # Allow your backend itself
    "*"  # (Not recommended for production, but useful for testing)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # 👈 Allow specific origins
    allow_credentials=True,
    allow_methods=["*"],  # 👈 Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # 👈 Allow all headers
)


app.include_router(predict.router)

@app.get("/")
def home():
    return {"message": "Welcome to the ML API"}
