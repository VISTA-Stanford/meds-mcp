#!/usr/bin/env bash
# Run the full fewshot_with_labels pipeline for the EHRSHOT readmission task
# using Vertex AI batch inference for all LLM steps.
#
# Steps:
#   0. Sample 100 stratified patients (50 readmitted / 50 not) from the test split
#   1. Precompute vignettes via Vertex batch (Gemini)
#   2. Build reason cache for train split via Vertex batch (readmission only)
#   3. Run baseline_vignette experiment via Vertex batch
#   4. Run vignette (fewshot) experiment via Vertex batch
#   5. Analyze results → balanced accuracy + flip/fix/hurt rates
#
# Usage:
#   bash experiments/fewshot_with_labels/run_readmission_vertex.sh
#
# Optional env overrides (all have sensible defaults from .env / _paths.py):
#   VERTEX_PROJECT   GCP project  (default: som-nero-plevriti-deidbdf)
#   VERTEX_LOCATION  GCP region   (default: us-central1)
#   MODEL            Gemini model (default: gemini-2.5-flash)
#   QUERY_SPLIT      Query split  (default: test)
#   SEED             Random seed  (default: 42)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Load .env so _paths.py picks up VISTA_COHORT_CSV / VISTA_CORPUS_DIR / VISTA_OUTPUTS_DIR
if [[ -f .env ]]; then
  set -a && source .env && set +a
fi

VERTEX_PROJECT="${VERTEX_PROJECT:-som-nero-plevriti-deidbdf}"
VERTEX_LOCATION="${VERTEX_LOCATION:-us-central1}"
MODEL="${MODEL:-gemini-2.5-flash}"
QUERY_SPLIT="${QUERY_SPLIT:-test}"
SEED="${SEED:-42}"
TASK="guo_readmission"
N_PATIENTS=100

GCS_BASE="gs://vista_bench/temp/pinnacle_templated_summaries"

OUTPUTS_DIR="${VISTA_OUTPUTS_DIR:-experiments/fewshot_with_labels/outputs}"
POOL_FILE="$OUTPUTS_DIR/pool_${QUERY_SPLIT}_${N_PATIENTS}.json"

echo "===== EHRSHOT readmission experiment ====="
echo "  task:         $TASK"
echo "  n_patients:   $N_PATIENTS (stratified 50/50 by readmission label)"
echo "  query_split:  $QUERY_SPLIT"
echo "  model:        $MODEL"
echo "  vertex:       $VERTEX_PROJECT / $VERTEX_LOCATION"
echo "  outputs_dir:  $OUTPUTS_DIR"
echo

# ---------------------------------------------------------------------------
# Step 0: Sample evaluation pool (stratified by readmission label)
# ---------------------------------------------------------------------------
echo "[0/5] Sampling ${N_PATIENTS} stratified patients (task=${TASK}, split=${QUERY_SPLIT})"
uv run python experiments/fewshot_with_labels/sample_pool.py \
  --split "$QUERY_SPLIT" \
  --n "$N_PATIENTS" \
  --seed "$SEED" \
  --stratify-task "$TASK" \
  --require-vignette \
  --require-item

echo "  Pool written to: $POOL_FILE"
echo

# ---------------------------------------------------------------------------
# Step 1: Precompute vignettes (Vertex batch)
# ---------------------------------------------------------------------------
echo "[1/5] Precomputing vignettes via Vertex batch"
uv run python experiments/fewshot_with_labels/precompute_vignettes_vertex_batch.py \
  --model "$MODEL" \
  --vertex-project "$VERTEX_PROJECT" \
  --vertex-location "$VERTEX_LOCATION" \
  --vertex-input-uri  "${GCS_BASE}/input/readmission_vignette_batch.jsonl" \
  --vertex-output-prefix "${GCS_BASE}/output/readmission_vignette_batch"
echo

# ---------------------------------------------------------------------------
# Step 2: Build reason cache for train split (readmission only, Vertex batch)
# ---------------------------------------------------------------------------
echo "[2/5] Building reason cache for task=${TASK} (train split, Vertex batch)"
uv run python experiments/fewshot_with_labels/build_reason_cache.py \
  --mode vertex_batch \
  --split train \
  --tasks "$TASK" \
  --model "$MODEL" \
  --vertex-project "$VERTEX_PROJECT" \
  --vertex-location "$VERTEX_LOCATION" \
  --vertex-input-uri  "${GCS_BASE}/input/readmission_reason_cache.jsonl" \
  --vertex-output-prefix "${GCS_BASE}/output/readmission_reason_cache"
echo

# ---------------------------------------------------------------------------
# Step 3: Run baseline_vignette experiment (no fewshot examples)
# ---------------------------------------------------------------------------
echo "[3/5] Running baseline_vignette experiment via Vertex batch"
uv run python experiments/fewshot_with_labels/run_experiment_vertex_batch.py \
  --context baseline_vignette \
  --pool "$POOL_FILE" \
  --query-split "$QUERY_SPLIT" \
  --tasks "$TASK" \
  --model "$MODEL" \
  --vertex-project "$VERTEX_PROJECT" \
  --vertex-location "$VERTEX_LOCATION" \
  --vertex-input-uri  "${GCS_BASE}/input/readmission_exp_baseline.jsonl" \
  --vertex-output-prefix "${GCS_BASE}/output/readmission_exp_baseline"
echo

# ---------------------------------------------------------------------------
# Step 4: Run vignette (fewshot) experiment
# ---------------------------------------------------------------------------
echo "[4/5] Running vignette (fewshot) experiment via Vertex batch"
uv run python experiments/fewshot_with_labels/run_experiment_vertex_batch.py \
  --context vignette \
  --pool "$POOL_FILE" \
  --query-split "$QUERY_SPLIT" \
  --tasks "$TASK" \
  --top-k 3 \
  --model "$MODEL" \
  --vertex-project "$VERTEX_PROJECT" \
  --vertex-location "$VERTEX_LOCATION" \
  --vertex-input-uri  "${GCS_BASE}/input/readmission_exp_fewshot.jsonl" \
  --vertex-output-prefix "${GCS_BASE}/output/readmission_exp_fewshot"
echo

# ---------------------------------------------------------------------------
# Step 5: Analyze results
# ---------------------------------------------------------------------------
echo "[5/5] Analyzing results"
uv run python experiments/fewshot_with_labels/analyze_results.py \
  --input-dir "$OUTPUTS_DIR" \
  --output-dir "$OUTPUTS_DIR"
echo
echo "===== Done ====="
echo "  analysis_summary.json  -> $OUTPUTS_DIR/analysis_summary.json"
echo "  per_task_accuracy.csv  -> $OUTPUTS_DIR/per_task_accuracy.csv"
