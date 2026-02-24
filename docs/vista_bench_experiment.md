# Vista Bench Experiment

This document describes how to run the vista_bench experiment comparing **LLM only** vs **LLM + tool** for each task.

## Overview

Context is restricted to **single-visit** by default: only events from the encounter containing `prediction_time` are included.

For each row in each task's label CSV:

1. **Load single-visit events** (precomputed cache or parsed from XML)
2. **LLM only**: Ask the question with `use_tools=False`
3. **LLM + tool**: Ask the question with `use_tools=True` and only the task-specific tool
4. **Record** raw and normalized responses with ground truth

**Precomputing** events once before the experiment avoids per-request XML parsing and speeds up API calls.

The tool is framed as **support**—the LLM is instructed to use its own judgment after seeing the tool result, not to follow it blindly.

## Prerequisites

- Patient XML files in the corpus directory (from `configs/vista.yaml`: `data/collections/vista_bench/thoracic_cohort_lumia`)
- **Only the subset of patients in the label CSVs are loaded**—not the full corpus. This speeds up startup.
- Label CSVs in `data/collections/vista_bench/labels/` (e.g. `labels_readmission.csv`, `labels_hyperkalemia.csv`, etc.)
- Network access for the LLM API (Stanford secure-llm)

## Running the Experiment

### Full experiment (all 14 tasks)

```bash
uv run python scripts/run_vista_bench_experiment.py --config configs/vista.yaml
```

### Precompute single-visit events first (faster API calls)

```bash
uv run python scripts/run_vista_bench_experiment.py --config configs/vista.yaml --precompute
```

Or run precompute separately:

```bash
uv run python scripts/precompute_single_visit_events.py --config configs/vista.yaml
# Saves to results/single_visit_events_cache.json
```

### Single task with limit (quick test)

```bash
uv run python scripts/run_vista_bench_experiment.py \
  --config configs/vista.yaml \
  --task guo_readmission \
  --limit 5
```

### Custom output path

```bash
uv run python scripts/run_vista_bench_experiment.py \
  --config configs/vista.yaml \
  --output results/my_run.jsonl
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--config` | `configs/vista.yaml` | Server config (defines `corpus_dir`, `load_all_patients`, etc.) |
| `--task` | (all tasks) | Run only this task (e.g. `guo_readmission`, `lab_hyperkalemia`) |
| `--limit` | (no limit) | Max rows per task (for quick testing) |
| `--output` | `results/vista_bench_<timestamp>.jsonl` | Output file path |
| `--output-dir` | `results` | Directory for output when `--output` is not set |
| `--precompute` | false | Run single-visit event precomputation before the experiment |
| `--precomputed-events-path` | `output-dir/single_visit_events_cache.json` | Path to precomputed events cache |
| `--delay-seconds` | 0.5 | Throttle: seconds to wait between patients |
| `--resume` | false | Resume from existing output (skip already-processed rows). Requires `--output`. |
| `--max-retries` | 5 | Max retries per LLM call on 429/token limit |
| `--batch-size` | 5 | Max concurrent API calls for parallelization |

## Tasks

**Binary tasks** (response: yes/no):  
`new_celiac`, `guo_icu`, `guo_los`, `new_lupus`, `new_acutemi`, `new_pancan`, `guo_readmission`, `new_hyperlipidemia`, `new_hypertension`

**Categorical tasks** (response: severe/moderate/mild/normal):  
`lab_anemia`, `lab_hyperkalemia`, `lab_hypoglycemia`, `lab_hyponatremia`, `lab_thrombocytopenia`

## Output Files

- **Results JSONL** (`results/vista_bench_<timestamp>.jsonl`): One JSON object per patient–task.
- **Summary JSON** (`results/vista_bench_<timestamp>_summary.json`): Per-task patient count, elapsed time, and total run time.

## Output Format

Each line in the JSONL is a JSON object:

```json
{
  "patient_id": "136038415",
  "task_name": "lab_hyperkalemia",
  "prediction_time": "2023-05-02T11:45:00.000000",
  "ground_truth_raw": "2",
  "ground_truth_normalized": "moderate",
  "llm_only_raw": "Based on the timeline, the severity is moderate.",
  "llm_only_normalized": "moderate",
  "llm_plus_tool_raw": "The tool indicates moderate; I agree.",
  "llm_plus_tool_normalized": "moderate"
}
```

The summary file format:

```json
{
  "output_file": "results/vista_bench_20250210_143052.jsonl",
  "total_patients": 5000,
  "num_patients_loaded": 450,
  "total_elapsed_seconds": 3600.5,
  "tasks": {
    "guo_readmission": {"num_patients": 607, "elapsed_seconds": 420.3},
    "lab_hyperkalemia": {"num_patients": 864, "elapsed_seconds": 612.1}
  }
}
```

## Rate limiting and resume

- **429 / Token limit**: The script retries with backoff (parses "try again in X seconds" from error) up to `--max-retries` (default 5).
- **Throttle**: `--delay-seconds` (default 2) adds a pause between patients to reduce burst traffic.
- **Resume**: If the run fails mid-way, rerun with `--resume --output results/your_file.jsonl` to skip already-completed rows and append new ones.

## Environment Variables

- `VISTA_LABELS_DIR`: Override labels directory (default: `data/collections/vista_bench/labels`)
- `DATA_DIR`: Override corpus directory (default from config)
- `LOAD_ALL_PATIENTS`: Set to `true` to load all patients at startup (recommended for experiments)

## Balanced cohort (100 patients, 50/50) and single-turn tool

You can create a **balanced cohort** of 100 patients per task (50% positive, 50% negative for binary tasks; lab tasks binarized to normal vs abnormal then 50/50). The **tool arm** is **single-turn**: the task prediction is pre-fetched and injected into each patient’s context (“tool already called; result: X”), and the task tool is not exposed, so the model answers in one turn using the injected results.

1. **Create balanced cohort** (from full Vista labels):
   ```bash
   uv run python scripts/sample_task_labels.py \
     --input-dir data/collections/vista_bench/labels \
     --output-dir data/collections/vista_bench/labels_100 \
     --n 100 \
     --balanced \
     --seed 42
   ```

2. **Precompute single-visit events** for the cohort (so context is unchanged and fast):
   ```bash
   VISTA_LABELS_DIR=data/collections/vista_bench/labels_100 uv run python scripts/precompute_single_visit_events.py --config configs/vista.yaml
   ```
   Optionally precompute **formatted context** cache for fastest runs:
   ```bash
   VISTA_LABELS_DIR=data/collections/vista_bench/labels_100 uv run python scripts/precompute_context_cache.py --config configs/vista.yaml --output-dir results
   ```

3. **Run the experiment** using the balanced labels and single-turn tool:
   ```bash
   VISTA_LABELS_DIR=data/collections/vista_bench/labels_100 uv run python scripts/run_vista_bench_experiment.py --config configs/vista.yaml --precompute
   ```
   If you precomputed context cache, pass it so the script uses it:
   ```bash
   VISTA_LABELS_DIR=data/collections/vista_bench/labels_100 uv run python scripts/run_vista_bench_experiment.py --config configs/vista.yaml --context-cache-path results/context_cache.json
   ```
   Or use `--precompute-context` to run precompute_context_cache.py before the experiment (builds context cache from events or XML).

4. **Analyze results** (analysis treats lab tasks with binary labels as binary for AUROC/balance):
   ```bash
   uv run python scripts/analyze_vista_bench_results.py --input results/vista_bench_<timestamp>.jsonl --output-dir results
   ```

## Future: Evaluation

Use `ground_truth_normalized` to compare against `llm_only_normalized` and `llm_plus_tool_normalized` to compute accuracy and compare LLM vs LLM+tool performance.
