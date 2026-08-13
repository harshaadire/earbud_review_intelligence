#!/bin/bash
# Starts the FastAPI backend in the background, then the Streamlit dashboard
# in the foreground (keeps the container alive).

uvicorn api.main:app --host 0.0.0.0 --port 8000 &

# Give the API a moment to start loading the model before the dashboard
# starts making requests to it
sleep 5

streamlit run app/dashboard.py --server.port 8501 --server.address 0.0.0.0
