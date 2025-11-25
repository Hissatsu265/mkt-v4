#!/bin/bash

# Navigate to the script's directory
cd "$(dirname "$0")"

# Load environment variables from your .env file
export $(grep -v '^#' .env | xargs)

# Determine host and port from environment (default fallback)
HOST=${LOCAL_HOST:-0.0.0.0}
PORT=${LOCAL_PORT:-8000}


# Run the FastAPI app
uvicorn app.main:app --host "$HOST" --port "$PORT" --reload
