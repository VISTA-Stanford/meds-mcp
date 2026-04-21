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

**BM25 similarity is computed on vignettes only, never on raw timelines** — regardless of `--context`. One BM25 index per task:

- Candidates: patients in the `train` split with `label != -1` for that task.
- Each candidate's document is their **task-aligned vignette** — the `PatientState` at `(candidate.person_id, candidate_item.embed_time)`. When a patient has different embed_times across tasks, the vignette used for BM25 (and for the LLM context) is the one built at the embed_time matching the task being searched.
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

`analyze_results.py` uses these pairings when computing flip / fix / hurt rates. Cross-pair comparisons (`baseline_vignette` vs `baseline_timeline`, `vignette` vs `timeline`) isolate the effect of query representation alone.

Similar patients' question text is omitted (same task ⇒ identical wording, just wastes tokens). Label is shown as `Yes` / `No` (from `label_description`), not the numeric code.

### Evaluation defaults

- Query pool: sampled from a split (`valid` by default, `test` optional); retrieval candidates come only from `train`.
- LLM: `temperature=0`, seed logged per row. Output parsed to `Yes`/`No` and compared to ground truth.
- Per-task n may be small depending on pool size; `analyze_results.py` flags tasks with `n < 10`. **Primary metric: overall accuracy.**

---

## Setup

### Local

```bash
uv sync                               # installs Python deps from pyproject.toml + uv.lock
cp .env.example .env                  # then edit .env with your VAULT_SECRET_KEY
```

`.env` is auto-loaded on import by `experiments/fewshot_with_labels/_paths.py`. On a laptop with the repo's default data layout, no env vars need to be set.

### GCP VM (or any Debian/Ubuntu host)

The scripts use plain `pathlib`/`open()`, so `gs://...` URIs don't work directly. The setup script copies data from GCS to local disk and writes a `.env` pointing at the local copies — you only paste the vault key afterward.

```bash
git clone <repo-url> && cd meds-mcp

# Installs apt + uv, copies CSV and XML corpus from GCS to local disk,
# generates .env with the correct local paths. One command.
bash scripts/setup_vm.sh --copy-gcs
source ~/.bashrc

# Paste VAULT_SECRET_KEY into the generated .env — only manual step.
${EDITOR:-nano} .env

uv sync
```

**Auth:** `gsutil` (and the fallback gcsfuse) need read access to the source buckets. Either attach a service account with `storage.objectViewer` to the VM, or run `gcloud auth application-default login` first. Create the VM with `--scopes=cloud-platform` for the zero-config path.

**Defaults used by `--copy-gcs`:**

| | Source | Local destination |
|---|---|---|
| CSV | `gs://su_vista_scratch/bikia_dev/lumia_cohort_progression_tasks-000000000000.csv` | `$HOME/data/lumia_cohort_progression_tasks-000000000000.csv` |
| XML corpus | `gs://vista_bench/thoracic_cohort_lumia/` | `$HOME/data/thoracic_cohort_lumia/` |
| Outputs | — | `$HOME/results/fewshot_with_labels_outputs/` |

Override any of them at call time:

```bash
GCS_COHORT_CSV_URI=gs://my-bucket/some.csv \
GCS_CORPUS_DIR_URI=gs://my-bucket/corpus \
LOCAL_DATA_ROOT=$HOME/scratch \
  bash scripts/setup_vm.sh --copy-gcs
```

Rerunning the script is safe: the CSV download skips if already present, and `gsutil -m rsync -r` for the corpus only transfers new/changed files.

**Alternative: `--mount-gcs`** — if you prefer gcsfuse mounts over copying (useful when the corpus is too big to host locally). Same one-command flow, generates a `.env` pointing at `$HOME/gcs/<bucket>/...`. Noticeably slower for the 8k-XML precompute step than local disk, though.

### Path resolution precedence (highest → lowest)

1. CLI flag (`--csv`, `--corpus-dir`, `--output-dir`, `--patients`, `--items`, `--pool`).
2. Env var already exported in the shell.
3. `.env` at repo root (auto-loaded).
4. Built-in local defaults (`data/collections/vista_bench/...`, `experiments/fewshot_with_labels/outputs/`).

| Env var | What it points at |
|---|---|
| `VISTA_COHORT_CSV` | Source CSV for `build_cohort.py` |
| `VISTA_CORPUS_DIR` | Directory of per-patient `{pid}.xml` files |
| `VISTA_OUTPUTS_DIR` | Where all experiment artifacts are written |
| `VAULT_SECRET_KEY` | Secure-llm auth (read by `securellm.Client` directly) |

---

## Run — exact commands (from repo root)

### 1. Build cohort (fast, no LLM)

```bash
uv run python experiments/fewshot_with_labels/build_cohort.py
# -> outputs/patients.jsonl  (one PatientState per (pid, embed_time))
# -> outputs/items.jsonl     (one LabeledItem per (pid, task), label != -1)
```

### 2. Precompute vignettes (~8k LLM calls; resumable)

```bash
# Default: all events on/before embed_time (--n-encounters 0), USMLE-style prompt.
uv run python experiments/fewshot_with_labels/precompute_vignettes.py \
  --model apim:gpt-4.1-mini
```

Rerun the same command to resume after a crash; it skips states that already have a vignette. Add `--force` to regenerate all vignettes (e.g. after changing `configs/prompts/vignette_prompt.txt`). Use `--limit 30` for a small smoke test before the full run.

### 3. Sample the evaluation pool

Pick one pattern depending on what you want to evaluate on:

```bash
# A) Pilot: 100 random pids from valid (seeded, reproducible).
uv run python experiments/fewshot_with_labels/sample_pool.py \
  --split valid --n 100 --seed 42 --require-vignette --require-item
# -> outputs/pool_valid_100.json

# B) Full test-split run: every eligible test patient (no sampling).
uv run python experiments/fewshot_with_labels/sample_pool.py \
  --split test --all --require-vignette --require-item
# -> outputs/pool_test_all.json
```

`--require-vignette` drops patients whose state has no vignette (e.g. `embed_time` before their first XML event). `--require-item` drops patients with zero non-(−1) items.

### 4. Run the experiment — one invocation per `--context`

```bash
# Change --pool and --query-split to match the pool you sampled in step 3.
POOL=experiments/fewshot_with_labels/outputs/pool_test_all.json
QSPLIT=test

uv run python experiments/fewshot_with_labels/run_experiment.py \
  --context baseline_vignette --pool $POOL --query-split $QSPLIT \
  --top-k 3 --model apim:gpt-4.1-mini

uv run python experiments/fewshot_with_labels/run_experiment.py \
  --context vignette          --pool $POOL --query-split $QSPLIT \
  --top-k 3 --model apim:gpt-4.1-mini

uv run python experiments/fewshot_with_labels/run_experiment.py \
  --context baseline_timeline --pool $POOL --query-split $QSPLIT \
  --top-k 3 --model apim:gpt-4.1-mini

uv run python experiments/fewshot_with_labels/run_experiment.py \
  --context timeline          --pool $POOL --query-split $QSPLIT \
  --top-k 3 --model apim:gpt-4.1-mini
```

The scripts skip the BM25 build entirely for `baseline_*` variants (no similars shown → no retrieval). Each `--context` writes its own `experiment_results_<context>.jsonl` and can be rerun independently.

### 5. Aggregate + compare to ground truth

```bash
uv run python experiments/fewshot_with_labels/analyze_results.py
# -> outputs/analysis_summary.json    per-variant accuracy, per-task matrix, paired flip/fix/hurt
# -> outputs/per_task_accuracy.csv    wide table for spreadsheet import
# also prints a human-readable per-task table and paired-comparison block to stdout
```

---

## Inspection utilities

### Preview the prompt (no LLM call)

```bash
# Pick first pool pid + first non-(-1) task, print all 4 context prompts:
uv run python experiments/fewshot_with_labels/show_prompt.py --context all

# Specific (pid, task), single context:
uv run python experiments/fewshot_with_labels/show_prompt.py \
  --person-id 135908791 --task died_any_cause_1_yr --context vignette
```

### Sanity-check a results file

```bash
# Per-context row count and parse rate:
for ctx in baseline_vignette vignette baseline_timeline timeline; do
  f=experiments/fewshot_with_labels/outputs/experiment_results_$ctx.jsonl
  [ -f "$f" ] || continue
  n=$(wc -l < "$f")
  p=$(grep -c '"pred": "\(Yes\|No\)"' "$f")
  echo "$ctx: $p/$n parsed"
done
```

---

## Outputs

Default: `experiments/fewshot_with_labels/outputs/` (or `$VISTA_OUTPUTS_DIR`). Directory is gitignored.

| File | Step | Description |
|---|---|---|
| `patients.jsonl` | 1, 2 | `PatientState` per `(person_id, embed_time)`; vignette filled by step 2 |
| `items.jsonl` | 1 | `LabeledItem` per `(person_id, task)` after dropping `label=-1` |
| `pool_<split>_<tag>.json` | 3 | Sampled evaluation pool (sorted list of person_ids) |
| `pool_<split>_<tag>_manifest.json` | 3 | Sampling config (split, seed, sizes, whether `--all` was used) |
| `experiment_results_<context>.jsonl` | 4 | One row per (pid, task): context, pred, raw, label, similars, model/seed/temperature |
| `experiment_meta_<context>.json` | 4 | Run settings |
| `analysis_summary.json` | 5 | Per-variant accuracy, per-task matrix, paired flip/fix/hurt |
| `per_task_accuracy.csv` | 5 | Flat per-task × per-variant table for spreadsheets |

---

## CLI reference

### `build_cohort.py`
- `--csv` (default: `VISTA_COHORT_CSV` → local CSV) — source of rows.
- `--output-dir` (default: `VISTA_OUTPUTS_DIR`) — where to write patients/items.
- `--drop-label` — labels to drop (repeatable; default `-1`).

### `precompute_vignettes.py`
- `--patients` / `--items` — cohort JSONLs (defaults honor env vars).
- `--corpus-dir` (default: `VISTA_CORPUS_DIR`) — XML directory.
- `--n-encounters` (default: **0** = all events on/before `embed_time`).
- `--model` (default: `apim:gpt-4.1-mini`).
- `--limit N` — process only N states (smoke).
- `--force` — regenerate even if vignette already present.
- `--no-progress` — disable tqdm.

### `sample_pool.py`
- `--split` (default `valid`).
- `--n N` (default 100) — or `--all` to include every eligible pid.
- `--seed` (default 42; ignored under `--all`).
- `--require-vignette` / `--require-item` — skip empty-vignette or no-item patients.

### `run_experiment.py`
- `--context {baseline_vignette,baseline_timeline,vignette,timeline}` *(required)*.
- `--pool` — JSON list of pids (produced by step 3).
- `--query-split` (default `valid`) / `--candidate-split` (default `train`).
- `--top-k` (default 3) — similar patients injected into the prompt.
- `--n` — process only first N pids in the pool (debug).
- `--n-encounters` (default 0) — capacity on timeline context blocks.
- `--max-chars` (default 120_000) — per-block timeline truncation.
- `--model`, `--seed`, `--temperature`, `--delay-seconds`.

### `analyze_results.py`
- `--input-dir` / `--output-dir` (defaults: `VISTA_OUTPUTS_DIR`).

### `show_prompt.py`
- `--person-id`, `--task`, `--context` (or `all`), `--top-k`, `--show-system`.

---

## Defensive invariants

- `TaskAwareRetriever.retrieve` asserts every returned neighbor has `split == "train"` (or whatever `--candidate-split` was set to) and `label not in drop_labels`.
- BM25 is invariantly vignette↔vignette — comment at `run_experiment.py` call site and `task_retriever.py` module docstring both note this.
- `CohortStore` rejects duplicate `(person_id, embed_time)` keys when constructed.
