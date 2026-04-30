#!/usr/bin/env python3
"""
Minimal LLM-only experiment: lab_thrombocytopenia only, no tools.

Calls the secure-llm API directly (no cohort_chat, no tools, no tool loop).
Uses context_cache.json and the same prompt text as cohort_chat for Vista task + use_tools=False.
When context cache is provided, the document store is not initialized (rows not in cache are skipped).

Usage:
  uv run python scripts/run_minimal_llm_only.py --config configs/vista.yaml --context-cache-path results/context_cache.json
  uv run python scripts/run_minimal_llm_only.py --config configs/vista.yaml --precompute-context --limit 10
  uv run python scripts/run_minimal_llm_only.py --config configs/vista.yaml --context-cache-path results/context_cache.json --limit 5 --output results/minimal_llm_only.jsonl
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from meds_mcp.experiments.minimal_LLM_only.run import main

if __name__ == "__main__":
    main()
