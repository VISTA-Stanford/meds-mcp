# Minimal LLM-only experiment

Single-task (lab_thrombocytopenia), LLM-only experiment with **no tools**. Calls the **secure-llm API directly** (no cohort_chat, no tools, no tool loop). Uses the same context cache and prompt text as cohort_chat for Vista task + use_tools=False.

## Prerequisites

- **context_cache.json** built for lab_thrombocytopenia (e.g. from `precompute_context_cache.py`).
- Label CSV for lab_thrombocytopenia in the labels dir (see `VISTA_LABELS_DIR` / config `labels_dir`).
- Network access for the LLM API.

## Usage

**With existing context cache:**

```bash
uv run python scripts/run_minimal_llm_only.py --config configs/vista.yaml --context-cache-path results/context_cache.json
```

**Precompute context then run (lab_thrombocytopenia only):**

```bash
uv run python scripts/run_minimal_llm_only.py --config configs/vista.yaml --precompute-context --limit 10
```

**Options:**

- `--context-cache-path PATH` – Path to precomputed `context_cache.json`. Default: `output-dir/context_cache.json`.
- `--precompute-context` – Run `precompute_context_cache.py` for lab_thrombocytopenia before the experiment.
- `--limit N` – Limit number of rows (for quick tests).
- `--output PATH` – Output JSONL path.
- `--output-dir DIR` – Output directory (default: `results`).
- `--resume` – Resume from existing output file (requires `--output`).
- `--batch-size N` – Max concurrent API calls (default: 1).
- `--no-cache` – Disable response caching.
- `--model NAME` – LLM model name.

## Output

- **JSONL** – One record per row: `patient_id`, `task_name`, `prediction_time`, `ground_truth_*`, `llm_only_raw`, `llm_only_normalized`.
- **Summary JSON** – `*_summary.json` with total patients, elapsed time, task name.

Rows not present in the context cache are skipped (no document store init when using the cache).
