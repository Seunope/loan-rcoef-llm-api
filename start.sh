#!/bin/bash

# Install Conda dependencies from environment.yml
conda env create -f environment.yml || conda env update -f environment.yml

# Activate the Conda environment
source activate rcoef-api 

# Start FastAPI server
uvicorn main:app --host 0.0.0.0 --port $PORT
