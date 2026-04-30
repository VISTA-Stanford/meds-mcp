#!/usr/bin/env bash
# Run baseline_lumia + lumia (few-shot) for the existing 100-patient readmission cohort.
# Uses filtered+truncated LUMIA timelines as context (lumia_filter.py).
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
MAX_CHARS="${MAX_CHARS_PER_PATIENT:-80000}"
GCS_INPUT="gs://vista_bench/temp/pinnacle_templated_summaries/input/ehrshot_lumia_batch.jsonl"
GCS_OUTPUT="gs://vista_bench/temp/pinnacle_templated_summaries/output/ehrshot_lumia_batch"

echo "===== LUMIA timeline experiment (baseline + fewshot) ====="
echo "  pool:              $POOL"
echo "  model:             $MODEL"
echo "  top_k:             $TOP_K"
echo "  max_chars/patient: $MAX_CHARS"
echo ""

echo "[1/3] Zero-shot baseline (baseline_lumia)"
uv run python experiments/fewshot_with_labels/run_lumia_experiment_vertex_batch.py \
  --context baseline_lumia \
  --patients "$PATIENTS" \
  --items "$ITEMS" \
  --pool "$POOL" \
  --corpus-dir "$LUMIA_XML_DIR" \
  --output-dir "$OUT_DIR" \
  --query-split test \
  --tasks guo_readmission \
  --model "$MODEL" \
  --max-chars-per-patient "$MAX_CHARS" \
  --vertex-input-uri "$GCS_INPUT" \
  --vertex-output-prefix "$GCS_OUTPUT"
echo ""

echo "[2/3] Few-shot (lumia, top-k=$TOP_K)"
uv run python experiments/fewshot_with_labels/run_lumia_experiment_vertex_batch.py \
  --context lumia \
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
  --max-chars-per-patient "$MAX_CHARS" \
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
