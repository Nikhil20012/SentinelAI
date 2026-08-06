#!/bin/bash
# Set up the development environment for SentinelAI.
# Run from the project root: bash scripts/setup.sh

set -e

echo "Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

echo "Installing sentinel-ai in editable mode with dev dependencies..."
pip install -e ".[dev]"

echo "Running initial tests..."
pytest tests/ -v

echo ""
echo "Setup complete. Activate with: source .venv/bin/activate"
