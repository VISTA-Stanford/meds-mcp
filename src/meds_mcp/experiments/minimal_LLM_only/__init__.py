"""
Minimal LLM-only experiment: single task (lab_thrombocytopenia), no tools.

Uses context_cache.json and the same preprocessing as the full vista_bench experiment.
Run via: uv run python scripts/run_minimal_llm_only.py --config configs/vista.yaml --context-cache-path results/context_cache.json
"""

from meds_mcp.experiments.minimal_LLM_only.run import main

__all__ = ["main"]
