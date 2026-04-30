#!/usr/bin/env bash
# Run the full EHRSHOT readmission experiment using LUMIA XML timelines.
#
# Pipeline (6 steps):
#   Step 1 — build_cohort:       Convert EHRSHOT CSVs → patients.jsonl + items.jsonl
#   Step 2 — precompute_vignette: Generate USMLE vignettes from LUMIA XML via Vertex batch
#   Step 3 — sample_pool:        Stratified 50+50 test patients (by readmission label)
#   Step 4 — zero-shot baseline: Run baseline_vignette context (no neighbors in prompt)
#   Step 5 — few-shot:           Run vignette context (top-k similar train patients shown)
#   Step 6 — analyze:            Balanced accuracy + flip/fix/hurt rates → JSON + CSV
#
# NOTE: The reason-cache step is intentionally skipped (code preserved in build_reason_cache.py).
#   To re-enable: add step 4 calling build_reason_cache.py and pass --reason-cache / --reason-missing-policy
#   to run_experiment_vertex_batch.py (remove --reason-missing-policy omit).
#
# Stratification: 50 readmitted (label=1) + 50 not readmitted (label=0)
#   selected from the TEST split. Train split is used in full for retrieval.
#
# Required env:
#   LUMIA_XML_DIR   Path to directory containing per-patient {pid}.xml LUMIA timeline files
#
# Optional env overrides:
#   EHRSHOT_ASSETS      Path to EHRSHOT_ASSETS directory (default: ~/LRRL_MEDS/data/EHRSHOT_ASSETS)
#   N_PER_LABEL         Patients per label class (default: 50 → 100 total)
#   TOP_K               Few-shot neighbors (default: 3)
#   SEED                Random seed (default: 42)
#   MODEL               Gemini model name (default: gemini-2.5-flash)
#   CANDIDATE_SPLIT     Split used as few-shot retrieval pool (default: test = within-cohort).
#                       Set to "train" to retrieve from the full train split instead.
#   VIGNETTE_PROMPT     Path to a custom vignette system prompt file for precompute step.
#
# Usage:
#   LUMIA_XML_DIR=/path/to/lumia/xml bash experiments/fewshot_with_labels/run_readmission_ehrshot_vertex.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# ---- Configuration --------------------------------------------------------
TASK="guo_readmission"
N_PER_LABEL="${N_PER_LABEL:-50}"
TOP_K="${TOP_K:-3}"
SEED="${SEED:-42}"
MODEL="${MODEL:-gemini-2.5-flash}"
CANDIDATE_SPLIT="${CANDIDATE_SPLIT:-test}"   # "test" = within-cohort (Option B); "train" = train-pool retrieval
VIGNETTE_PROMPT="${VIGNETTE_PROMPT:-$REPO_ROOT/configs/prompts/vignette_prompt_readmission.txt}"

EHRSHOT_ASSETS="${EHRSHOT_ASSETS:-$HOME/LRRL_MEDS/data/EHRSHOT_ASSETS}"
LABELS_CSV="$EHRSHOT_ASSETS/benchmark/guo_readmission/labeled_patients.csv"
SPLITS_CSV="$EHRSHOT_ASSETS/splits/person_id_map.csv"

if [[ -z "${LUMIA_XML_DIR:-}" ]]; then
  echo "ERROR: LUMIA_XML_DIR is not set. Point it to the directory of per-patient {pid}.xml files."
  echo "  Example: LUMIA_XML_DIR=/path/to/lumia_xml bash $0"
  exit 1
fi

OUT_DIR="$HOME/meds-mcp/experiments/fewshot_with_labels/outputs/ehrshot"
LOG_DIR="$HOME/meds-mcp/experiments/fewshot_with_labels/logs"
mkdir -p "$LOG_DIR"
RUN_LOG="$LOG_DIR/readmission_ehrshot_$(date +%s).log"

PATIENTS="$OUT_DIR/patients.jsonl"
ITEMS="$OUT_DIR/items.jsonl"
# REASON_CACHE="$OUT_DIR/reason_cache.jsonl"  # uncomment when reason cache step is re-enabled

# GCS paths (shared across steps — safe because each step uses a distinct object key)
GCS_INPUT_VIGNETTE="gs://vista_bench/temp/pinnacle_templated_summaries/input/ehrshot_vignette_batch.jsonl"
GCS_OUTPUT_VIGNETTE="gs://vista_bench/temp/pinnacle_templated_summaries/output/ehrshot_vignette_batch"
GCS_INPUT_EXP="gs://vista_bench/temp/pinnacle_templated_summaries/input/ehrshot_exp_batch.jsonl"
GCS_OUTPUT_EXP="gs://vista_bench/temp/pinnacle_templated_summaries/output/ehrshot_exp_batch"

log() { echo "$*" | tee -a "$RUN_LOG"; }

log "===== EHRSHOT readmission experiment ====="
log "  task:          $TASK"
log "  n_per_label:   $N_PER_LABEL  ($(( N_PER_LABEL * 2 )) total: ${N_PER_LABEL} readmitted + ${N_PER_LABEL} not)"
log "  query_split:   test   (train used in full for retrieval)"
log "  top_k:         $TOP_K"
log "  seed:          $SEED"
log "  model:         $MODEL"
log "  lumia_xml_dir: $LUMIA_XML_DIR"
log "  output_dir:    $OUT_DIR"
log ""

# ---------------------------------------------------------------------------
# Step 1: Build CohortStore (patients.jsonl + items.jsonl) from EHRSHOT CSVs
# ---------------------------------------------------------------------------
log "[1/7] Building cohort from EHRSHOT CSVs"
mkdir -p "$OUT_DIR"
uv run python experiments/fewshot_with_labels/build_ehrshot_cohort.py \
  --labels-csv "$LABELS_CSV" \
  --splits-csv "$SPLITS_CSV" \
  --task "$TASK" \
  --output-dir "$OUT_DIR" \
  --splits test train \
  2>&1 | tee -a "$RUN_LOG"
log ""

# ---------------------------------------------------------------------------
# Step 2: Sample stratified 100-patient pool from test split
#   (done before vignette generation so we can scope generation to pool only)
# ---------------------------------------------------------------------------
log "[2/6] Sampling stratified test pool (${N_PER_LABEL} per label class)"
uv run python experiments/fewshot_with_labels/sample_pool.py \
  --patients "$PATIENTS" \
  --items "$ITEMS" \
  --output-dir "$OUT_DIR" \
  --split test \
  --stratify-task "$TASK" \
  --n-per-label "$N_PER_LABEL" \
  --seed "$SEED" \
  2>&1 | tee -a "$RUN_LOG"
# sample_pool writes pool_test_<n>.json — find it.
POOL_ACTUAL=$(ls "$OUT_DIR"/pool_test_*.json 2>/dev/null | grep -v manifest | sort | tail -1)
if [[ -z "${POOL_ACTUAL:-}" ]]; then
  log "ERROR: pool file not found in $OUT_DIR"
  exit 1
fi
log "  Using pool: $POOL_ACTUAL"
log ""

# ---------------------------------------------------------------------------
# Step 3: Precompute USMLE vignettes from LUMIA XML timelines via Vertex batch
#   When CANDIDATE_SPLIT=test (within-cohort), scope to the pool patients only.
#   When CANDIDATE_SPLIT=train, generate for all patients (pool + train retrieval pool).
# ---------------------------------------------------------------------------
log "[3/6] Precomputing vignettes from LUMIA XML (Vertex batch)"
_VIGNETTE_PROMPT_ARG=()
[[ -n "$VIGNETTE_PROMPT" ]] && _VIGNETTE_PROMPT_ARG=(--vignette-prompt "$VIGNETTE_PROMPT")
_PERSON_IDS_ARG=()
if [[ "$CANDIDATE_SPLIT" == "test" ]]; then
  _PERSON_IDS_ARG=(--person-ids-file "$POOL_ACTUAL")
  log "  Within-cohort mode: generating vignettes for pool patients only ($(wc -w < "$POOL_ACTUAL") ids)"
fi
uv run python experiments/fewshot_with_labels/precompute_vignettes_vertex_batch.py \
  --patients "$PATIENTS" \
  --items "$ITEMS" \
  --corpus-dir "$LUMIA_XML_DIR" \
  --model "$MODEL" \
  "${_VIGNETTE_PROMPT_ARG[@]}" \
  "${_PERSON_IDS_ARG[@]}" \
  --vertex-input-uri "$GCS_INPUT_VIGNETTE" \
  --vertex-output-prefix "$GCS_OUTPUT_VIGNETTE" \
  2>&1 | tee -a "$RUN_LOG"
log ""

# ---------------------------------------------------------------------------
# Step 4: Zero-shot baseline (baseline_vignette) — no fewshot neighbors
# ---------------------------------------------------------------------------
log "[4/6] Zero-shot baseline (baseline_vignette)"
uv run python experiments/fewshot_with_labels/run_experiment_vertex_batch.py \
  --context baseline_vignette \
  --patients "$PATIENTS" \
  --items "$ITEMS" \
  --pool "$POOL_ACTUAL" \
  --corpus-dir "$LUMIA_XML_DIR" \
  --output-dir "$OUT_DIR" \
  --query-split test \
  --candidate-split "$CANDIDATE_SPLIT" \
  --tasks "$TASK" \
  --model "$MODEL" \
  --reason-missing-policy omit \
  --vertex-input-uri "$GCS_INPUT_EXP" \
  --vertex-output-prefix "$GCS_OUTPUT_EXP" \
  2>&1 | tee -a "$RUN_LOG"
log ""

# ---------------------------------------------------------------------------
# Step 5: Few-shot (vignette) — top-k similar train patients shown
# ---------------------------------------------------------------------------
log "[5/6] Few-shot (vignette, top-k=$TOP_K)"
uv run python experiments/fewshot_with_labels/run_experiment_vertex_batch.py \
  --context vignette \
  --patients "$PATIENTS" \
  --items "$ITEMS" \
  --pool "$POOL_ACTUAL" \
  --corpus-dir "$LUMIA_XML_DIR" \
  --output-dir "$OUT_DIR" \
  --query-split test \
  --candidate-split "$CANDIDATE_SPLIT" \
  --tasks "$TASK" \
  --top-k "$TOP_K" \
  --model "$MODEL" \
  --reason-missing-policy omit \
  --vertex-input-uri "$GCS_INPUT_EXP" \
  --vertex-output-prefix "$GCS_OUTPUT_EXP" \
  2>&1 | tee -a "$RUN_LOG"
log ""

# ---------------------------------------------------------------------------
# Step 6: Analyze — balanced accuracy + fewshot lift
# ---------------------------------------------------------------------------
log "[6/6] Analyzing results"
uv run python experiments/fewshot_with_labels/analyze_results.py \
  --input-dir "$OUT_DIR" \
  --output-dir "$OUT_DIR" \
  2>&1 | tee -a "$RUN_LOG"
log ""

log "===== Done ====="
log "  Log:              $RUN_LOG"
log "  analysis_summary: $OUT_DIR/analysis_summary.json"
log "  per_task_csv:     $OUT_DIR/per_task_accuracy.csv"
log "  example prompts:  $OUT_DIR/example_prompt_*.txt"
log ""
log "To re-enable reason cache: uncomment step 4 in this script and add --reason-cache / remove --reason-missing-policy omit from steps 4-5."
