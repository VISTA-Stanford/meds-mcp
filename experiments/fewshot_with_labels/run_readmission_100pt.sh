#!/usr/bin/env bash
# Re-run baseline + fewshot for the existing 100-patient readmission cohort.
# Skips steps 1-3 (cohort build, pool sampling, vignette precompute).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

OUT_DIR="$HOME/meds-mcp/experiments/fewshot_with_labels/outputs/ehrshot"
PATIENTS="$OUT_DIR/patients.jsonl"
ITEMS="$OUT_DIR/items.jsonl"
POOL="$OUT_DIR/pool_test_100.json"
LUMIA_XML_DIR="${LUMIA_XML_DIR:-/home/Ayeeshi/meds-mcp/data/ehrshot_lumia/meds_corpus}"
MODEL="${MODEL:-gemini-2.5-flash}"
TOP_K="${TOP_K:-3}"
GCS_INPUT="gs://vista_bench/temp/pinnacle_templated_summaries/input/ehrshot_exp_batch.jsonl"
GCS_OUTPUT="gs://vista_bench/temp/pinnacle_templated_summaries/output/ehrshot_exp_batch"

echo "===== Readmission 100pt re-run (baseline + fewshot) ====="
echo "  pool:   $POOL"
echo "  model:  $MODEL"
echo "  top_k:  $TOP_K"
echo ""

echo "[1/3] Zero-shot baseline (baseline_vignette)"
uv run python experiments/fewshot_with_labels/run_experiment_vertex_batch.py \
  --context baseline_vignette \
  --patients "$PATIENTS" \
  --items "$ITEMS" \
  --pool "$POOL" \
  --corpus-dir "$LUMIA_XML_DIR" \
  --output-dir "$OUT_DIR" \
  --query-split test \
  --candidate-split test \
  --tasks guo_readmission \
  --model "$MODEL" \
  --reason-missing-policy omit \
  --vertex-input-uri "$GCS_INPUT" \
  --vertex-output-prefix "$GCS_OUTPUT"
echo ""

echo "[2/3] Few-shot (vignette, top-k=$TOP_K)"
uv run python experiments/fewshot_with_labels/run_experiment_vertex_batch.py \
  --context vignette \
  --patients "$PATIENTS" \
  --items "$ITEMS" \
  --pool "$POOL" \
  --corpus-dir "$LUMIA_XML_DIR" \
  --output-dir "$OUT_DIR" \
  --query-split test \
  --candidate-split test \
  --tasks guo_readmission \
  --top-k "$TOP_K" \
  --model "$MODEL" \
  --reason-missing-policy omit \
  --vertex-input-uri "$GCS_INPUT" \
  --vertex-output-prefix "$GCS_OUTPUT"
echo ""

echo "[3/3] Analyzing results"
uv run python experiments/fewshot_with_labels/analyze_results.py \
  --input-dir "$OUT_DIR" \
  --output-dir "$OUT_DIR"
echo ""

echo "===== Done ====="
echo "  outputs: $OUT_DIR"
