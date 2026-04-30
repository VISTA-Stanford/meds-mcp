# EHRSHOT experiment

Binary tasks only; labels joined with MEDS patient IDs, one label per patient (prefer positive/abnormal), 50/50 cohort. Single-turn tool: result injected in context.

## Prerequisites

- EHRSHOT MEDS corpus: XML files per patient (e.g. under `data/collections/ehrshot/meds_corpus/`).
- Raw task label CSVs under `data/collections/ehrshot/labels/` (one CSV per task, same names as Vista: `labels_guo_icu.csv`, etc.).

Update `configs/ehrshot.yaml` if your paths differ (`data.corpus_dir`, `labels_dir`).

---

## 1. Create sample labels (join MEDS + one label per patient + 50/50)

From repo root:

```bash
uv run python scripts/sample_task_labels.py \
  --input-dir data/collections/ehrshot/labels \
  --output-dir data/collections/ehrshot/labels/labels_100 \
  --balanced \
  --meds-dir data/collections/ehrshot/meds_corpus \
  --one-label-per-patient \
  --n 100 \
  --seed 42
```

- `--meds-dir`: directory containing `*.xml` (stem = patient_id). Only patients in MEDS are kept.
- `--one-label-per-patient`: for each task, one row per patient; if any label is positive/abnormal use that, else use the given label.
- Output is written to `data/collections/ehrshot/labels/labels_100/` (50% positive, 50% negative per task).

Optional: `--meds-patient-ids-file path/to/ids.txt` (one patient_id per line) instead of `--meds-dir`.

---

## 2. Create context cache

Precompute formatted context for each (patient_id, prediction_time, task) so the experiment can run without loading the document store.

**Option A – from precomputed events (recommended)**

First precompute single-visit events:

```bash
uv run python scripts/precompute_single_visit_events.py \
  --config configs/ehrshot.yaml \
  --output-dir results
```

Then build the context cache from the events cache:

```bash
uv run python scripts/precompute_context_cache.py \
  --config configs/ehrshot.yaml \
  --events-cache results/single_visit_events_cache.json \
  --output-dir results
```

**Option B – from XML only**

```bash
uv run python scripts/precompute_context_cache.py \
  --config configs/ehrshot.yaml \
  --output-dir results
```

Output: `results/context_cache.json` (or `--output-dir`/context_cache.json). The config’s `labels_dir` is used so labels are read from `data/collections/ehrshot/labels/labels_100`.

---

## 3. Run the experiment (using cache and labels)

```bash
uv run python scripts/run_vista_bench_experiment.py \
  --config configs/ehrshot.yaml \
  --context-cache-path results/context_cache.json \
  --output-dir results
```

- Labels are read from the path in `configs/ehrshot.yaml` (`labels_dir`: `data/collections/ehrshot/labels/labels_100`).
- No document store init when `--context-cache-path` is set; rows not in the cache are skipped.
- Results: `results/vista_bench_<timestamp>.jsonl` and a summary JSON.

Optional: `--task guo_icu --limit 20` to run one task with a row limit; `--precompute-context` to run context cache build before the experiment.

---

## Debug: inspect context (no API call)

To see exactly what context is sent to the LLM for a small sample (no LLM call):

```bash
uv run python scripts/debug_context.py \
  --config configs/ehrshot.yaml \
  --context-cache results/context_cache.json \
  --task guo_icu \
  --limit 2
```

Prints the cohort context block and full user prompt for the first `--limit` rows of the task.
