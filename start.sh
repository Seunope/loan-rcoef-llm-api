#!/bin/bash

# Install dependencies using pip
pip install -r requirements.txt

# Start FastAPI app
uvicorn app.main:app --host 0.0.0.0 --port 8080
