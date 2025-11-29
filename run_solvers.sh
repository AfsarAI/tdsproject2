#!/bin/bash
# Kill existing processes to avoid conflicts
echo "Stopping existing solvers..."
pkill -f "app.py"
pkill -f "reference_repo/main.py"

# Path to venv python
VENV_PYTHON="./.venv/bin/python3"

# Check if venv exists
if [ ! -f "$VENV_PYTHON" ]; then
    echo "Error: Virtual environment not found at $VENV_PYTHON"
    exit 1
fi

# Start Reference Repo (Port 7860)
echo "Starting Reference Repo on Port 7860..."
if [ -f reference_repo/.env ]; then
    export $(grep -v '^#' reference_repo/.env | xargs)
fi
$VENV_PYTHON reference_repo/main.py > reference_logs.txt 2>&1 &
REF_PID=$!
echo "Reference Repo started with PID $REF_PID"

# Start User Project (Port 5000)
echo "Starting User Project on Port 5000..."
cd llm-quiz-solver
../$VENV_PYTHON app.py > ../user_logs.txt 2>&1 &
USER_PID=$!
cd ..
echo "User Project started with PID $USER_PID"

echo "---------------------------------------------------"
echo "BOTH SOLVERS ARE RUNNING!"
echo "1. User Project:      http://localhost:5000"
echo "2. Reference Project: http://localhost:7860"
echo "---------------------------------------------------"
echo "Logs are being written to user_logs.txt and reference_logs.txt"
