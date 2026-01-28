#!/bin/bash
# Test runner script that ensures the correct environment is used

set -e

# Check if we're in the project directory
if [ ! -f "pyproject.toml" ]; then
    echo "Error: Must be run from project root directory"
    exit 1
fi

# Ensure dependencies are installed
echo "Syncing dependencies..."
uv sync --extra dev --extra test

# Run tests using the venv's Python
echo "Running tests..."
uv run python -m pytest "$@"
