# Progression subset experiment

Evaluates three LLM setups on rows from `data/collections/vista_bench/progression_subset.csv`:

1. **v1** — Query patient timeline only (events on or before `embed_time`).
2. **v2** — v1 + similar patients (BM25 on vignettes); each similar’s context is chopped at **that patient’s** `embed_time`.
3. **v3** — v1 + similar patients; each similar’s context is chopped at the **query patient’s** `embed_time`.

The model must answer with exactly **-1**, **0**, or **1**. Analysis reports accuracy and flip/fix/hurt rates vs v1.

## Prerequisites

- Lumia XML under `data/collections/vista_bench/thoracic_cohort_lumia/` (from `configs/vista.yaml` `data.corpus_dir`).
- Secure LLM credentials (same as the rest of `meds-mcp`).
- BM25 retrieval is provided by `llama-index-retrievers-bm25` via `meds_mcp.similarity.PatientBM25Index`.

## Steps (from repository root)

```bash
# 1) Stratified sample: 100 rows, equal count per task
uv run python experiments/progression_subset/sample_cohort.py --n 100 --seed 42

# 2) Precompute vignettes (last N encounters after chop at each patient’s embed_time) + vignettes.jsonl
uv run python experiments/progression_subset/precompute_vignettes.py \
  --n-encounters 2 \
  --model apim:gpt-4.1-mini

# 3) Run experiment (three variants per sampled row)
uv run python experiments/progression_subset/run_experiment.py \
  --top-k 5 \
  --model apim:gpt-4.1-mini

# 4) Metrics
uv run python experiments/progression_subset/analyze_results.py
```

Outputs default to `experiments/progression_subset/outputs/`:

| File | Description |
|------|-------------|
| `sampled_rows.csv` | Cohort |
| `vignettes.jsonl` | One JSON object per patient with a vignette |
| `experiment_results.jsonl` | Predictions and raw LLM outputs |
| `analysis_summary.json` | Accuracy, flips, fix rates, hurts |

## Faster sampling

`sample_cohort.py` uses a two-pass reservoir sample (low memory). Plain `python3` works if the venv is not needed for that step.

## Options

- `precompute_vignettes.py`: `--limit K` to process only K patients (debug). Shows a **tqdm** progress bar by default (`--no-progress` to disable). Requires `tqdm` (listed in project dependencies). By default, **excludes any `person_id` that has at least one CSV row with a blank column** (so all columns must be populated on every row for that patient). Use `--no-require-complete-rows` to include everyone who appears in the CSV regardless of blank cells. Use **`--resume`** to continue after a stop: reads existing `vignettes.jsonl`, skips `person_id`s already present, and **appends** new rows (omit `--resume` to overwrite the file from scratch).
- `run_experiment.py`: `--limit N` for first N sampled rows; `--max-chars` to cap timeline size.
