#!/usr/bin/env bash
# Convenience script for running API + UI locally without Docker.
# Assumes you've already: pip install -r requirements.txt
# and either `ollama serve` (+ `ollama pull llama3.2:1b`) is running,
# or you've set LLM_PROVIDER=groq with GROQ_API_KEY in your .env.
set -euo pipefail

echo "Starting API on :8000 ..."
uvicorn app.main:app --host 0.0.0.0 --port 8000 &
API_PID=$!

trap "kill $API_PID" EXIT

sleep 2
echo "Starting UI on :8501 ..."
streamlit run ui/streamlit_app.py --server.address=0.0.0.0 --server.port=8501
