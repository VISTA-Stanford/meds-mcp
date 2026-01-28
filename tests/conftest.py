"""
Pytest configuration for meds-mcp tests.

This file ensures that the src directory is added to the Python path,
allowing tests to import the meds_mcp package even when pytest is run
directly (not through uv run).

Note: For best results, run tests using: uv run pytest
"""
import sys
import warnings
from pathlib import Path

# Add the src directory to Python path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Check if we're likely in the wrong environment
# If .venv exists but we're not using it, check for missing dependencies
venv_python = project_root / ".venv" / "bin" / "python"
if venv_python.exists() and sys.executable != str(venv_python):
    # Check if meilisearch is missing (common indicator of wrong environment)
    try:
        import meilisearch
    except ImportError:
        # Check if we're in a conda/base environment
        is_conda_env = "miniconda3" in sys.executable or "anaconda3" in sys.executable or "conda" in sys.executable.lower()
        if is_conda_env:
            warnings.warn(
                f"Warning: pytest is using {sys.executable} (conda environment) instead of the uv venv Python.\n"
                f"meilisearch and other dependencies are not available in this environment.\n"
                f"To fix this, run: uv sync --extra dev --extra test\n"
                f"Then use: uv run python -m pytest (or: uv run pytest)",
                UserWarning
            )
        else:
            warnings.warn(
                f"Warning: pytest is using {sys.executable} instead of the uv venv Python.\n"
                f"meilisearch (and other dependencies) are not available in this environment.\n"
                f"Please use 'uv run python -m pytest' to run tests with all dependencies available.\n"
                f"Alternatively, activate the venv: source .venv/bin/activate",
                UserWarning
            )
