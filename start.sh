#!/bin/bash

# Install dependencies using pip
pip install -r requirements.txt

# Start FastAPI app
uvicorn main:app --host 0.0.0.0 --port $PORT
