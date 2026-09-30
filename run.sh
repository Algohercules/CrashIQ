#!/usr/bin/env bash
# ==============================================================================
# CrashIQ Startup Script
# Automatically ensures dependencies, model artifacts, and launches the web app.
# ==============================================================================

set -e

# Navigate to the project root directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# 1. Activate or create virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Creating virtual environment..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
fi

# 2. Check if the trained model exists; if not, train it
if [ ! -f "models/best_model.pkl" ]; then
    echo "No trained model found. Running ML pipeline (main.py)..."
    python main.py
fi

# 3. Launch the Streamlit dashboard
echo "Launching CrashIQ Web Dashboard..."
echo "Access the app at: http://localhost:8501"
streamlit run app.py
