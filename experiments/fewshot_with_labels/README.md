# fewshot_with_labels experiment

Tests whether showing the LLM **similar patients' vignettes + their ground-truth Yes/No answers for the same task** improves binary predictions over a no-retrieval baseline. Retrieval is held fixed; only the level of context detail varies, selected at run time by `--context`.

## Design

### Data model (normalized)

Two JSONL-backed shapes. `PatientState` is keyed by `(person_id, embed_time)` — a patient may have several states when different tasks use different prediction times. `LabeledItem` carries its own `embed_time` and resolves to the corresponding state via that composite key.

- **`PatientState`** (one per unique `(person_id, embed_time)`): `split`, precomputed `vignette` built from the XML timeline chopped at `embed_time`, `created_at`.
- **`LabeledItem`** (one per `(person_id, task)` after dropping `label == -1`): `task`, `task_group`, `question`, `label`, `label_description`, `split`, `embed_time`, `created_at`.
- **`CohortItem`** (join view): `store.join(pid, task)` returns the state at `(pid, item.embed_time)` joined with the item.

Timelines are NOT stored — they are regenerated from the XML corpus at prompt-build time via `DeterministicTimelineLinearizationGenerator`. Precompute deduplicates by `(pid, embed_time)`, so two tasks sharing a prediction time share one vignette (no redundant LLM calls).

### Retrieval (fixed)

Per-task BM25 indices. For a query on task `X`:

- Candidates are **only patients in the train split with `label != -1` for task `X`**.
- Each candidate's document is their **task-aligned vignette** — the `PatientState` at `(candidate.person_id, candidate_item.embed_time)`. When a patient has different embed_times across tasks, the vignette used for BM25 (and later shown in the LLM context) is the one built at the embed_time matching the task being searched.
- The query's retrieval vignette is likewise the state at `(query.person_id, query_item.embed_time)`.

Implemented in `src/meds_mcp/similarity/task_retriever.py` (`TaskAwareRetriever`).

### LLM context (variable, one per run)

| `--context` | Query representation | Similars |
|---|---|---|
| `baseline_vignette` | Vignette | *(none)* |
| `vignette` | Vignette | Similar vignettes + Yes/No label for this task |
| `baseline_timeline` | Timeline | *(none)* |
| `timeline` | Timeline | Similar timelines (chopped at each similar's own `embed_time`) + Yes/No label |

The four variants form two paired 1:1 comparisons where only the similars block differs:

- `baseline_vignette` ↔ `vignette` — effect of adding similars, query rendered as vignette in both.
- `baseline_timeline` ↔ `timeline` — effect of adding similars, query rendered as timeline in both.

`analyze_results.py` uses these pairings when computing flip / fix / hurt rates. A cross-pair comparison (`baseline_vignette` vs `baseline_timeline`, or `vignette` vs `timeline`) isolates the effect of query representation alone.

Similar patients' question text is omitted (same task ⇒ identical wording, just wastes tokens). Label is shown as `Yes` / `No` (from `label_description`), not the numeric code.

### Evaluation

- Split policy: sample evaluation pool from the `valid` split; retrieval candidates come only from `train`. Test is held out.
- Query pool: seeded random sample of 100 unique `person_id` from `valid` (configurable via `--n` at sample time).
- For each sampled patient, run **all** their non-(-1) task items.
- LLM answer is parsed to `Yes` / `No` and compared to ground truth.

### Caveats

- 100 valid pids × 40 tasks — per-task n is small (order of tens). `analyze_results.py` flags tasks with `n < 10`. **Primary metric: overall accuracy.** Per-task numbers are exploratory.

---

## Prerequisites

- Lumia XML under `data/collections/vista_bench/thoracic_cohort_lumia/`.
- Full cohort CSV at `data/collections/vista_bench/bikia_dev-lumia_cohort_progression_tasks-000000000000.csv` (columns: `person_id, split, embed_time, task, task_group, question, label, label_description`).
- Secure LLM credentials (same as the rest of `meds-mcp`).

## Run (from repository root)

```bash
# 1) Build the cohort: filters label=-1, writes patients.jsonl + items.jsonl. Fast, no LLM.
uv run python experiments/fewshot_with_labels/build_cohort.py

# 2) Precompute vignettes for every patient (resumable; rerun to continue).
uv run python experiments/fewshot_with_labels/precompute_vignettes.py \
  --n-encounters 2 \
  --model apim:gpt-4.1-mini

# 3) Sample the evaluation pool (100 pids from valid split, seeded).
uv run python experiments/fewshot_with_labels/sample_pool.py \
  --n 100 --seed 42 --require-vignette --require-item

# 4) Run one context variant per invocation (repeat for each variant to compare).
uv run python experiments/fewshot_with_labels/run_experiment.py \
  --context baseline_vignette --top-k 3 --model apim:gpt-4.1-mini

uv run python experiments/fewshot_with_labels/run_experiment.py \
  --context vignette --top-k 3 --model apim:gpt-4.1-mini

uv run python experiments/fewshot_with_labels/run_experiment.py \
  --context timeline --top-k 3 --model apim:gpt-4.1-mini

# 5) Analyze (merges whichever experiment_results_<context>.jsonl files exist).
uv run python experiments/fewshot_with_labels/analyze_results.py
```

## Outputs

Default: `experiments/fewshot_with_labels/outputs/`

| File | Description |
|------|-------------|
| `patients.jsonl` | One `PatientState` per unique `(person_id, embed_time)` (vignette filled by precompute step) |
| `items.jsonl` | One `LabeledItem` per `(person_id, task)` after dropping label=-1 |
| `pool_valid_<n>.json` | Sampled evaluation pool (sorted list of person_ids) |
| `pool_valid_<n>_manifest.json` | Sampling manifest (seed, split, sizes) |
| `experiment_results_<context>.jsonl` | Per-row predictions for that run |
| `experiment_meta_<context>.json` | Run settings (model, seed, temperature, retrieval policy) |
| `analysis_summary.json` | Per-variant accuracy, per-task breakdown, vs-baseline flip/fix/hurt |

## CLI reference

### `run_experiment.py`

- `--context {baseline_vignette,baseline_timeline,vignette,timeline}` *(required)*
- `--top-k` (default 3)
- `--n` — process only the first N patients in the pool (debug); default all
- `--n-encounters` (default 2)
- `--seed` (default 42)
- `--temperature` (default 0.0)
- `--model` (default `apim:gpt-4.1-mini`)
- `--max-chars` (default 120_000) — per-block timeline truncation
- `--query-split` / `--candidate-split` (defaults `valid` / `train`)

### Defensive invariants

`TaskAwareRetriever.retrieve` asserts every returned neighbor has `split == "train"` and label not in `{-1}`. Cheap; catches cohort-rebuild bugs early.
