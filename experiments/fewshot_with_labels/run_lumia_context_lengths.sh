#!/usr/bin/env bash
# Sweep baseline_lumia across 4 context lengths: 4K, 8K, 16K, 32K chars per patient.
# Runs sequentially; each job writes experiment_results_baseline_lumia_{N}k.jsonl.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

OUT_DIR="$HOME/meds-mcp/experiments/fewshot_with_labels/outputs/ehrshot"
PATIENTS="$OUT_DIR/patients.jsonl"
ITEMS="$OUT_DIR/items.jsonl"
POOL="$OUT_DIR/pool_test_100.json"
LUMIA_XML_DIR="${LUMIA_XML_DIR:-/home/Ayeeshi/meds-mcp/data/ehrshot_lumia/meds_corpus}"
MODEL="${MODEL:-gemini-2.5-flash}"
GCS_INPUT="gs://vista_bench/temp/pinnacle_templated_summaries/input/ehrshot_lumia_batch.jsonl"
GCS_OUTPUT="gs://vista_bench/temp/pinnacle_templated_summaries/output/ehrshot_lumia_batch"

echo "===== LUMIA baseline context-length sweep ====="
echo "  context lengths: 4096 8192 16384 32768 chars/patient"
echo "  model: $MODEL"
echo "  pool:  $POOL"
echo ""

for CHARS in 4096 8192 16384 32768; do
    LABEL="baseline_lumia_${CHARS}"
    echo "--- [$(date +%H:%M:%S)] $LABEL (max_chars=$CHARS) ---"
    uv run python experiments/fewshot_with_labels/run_lumia_experiment_vertex_batch.py \
      --context baseline_lumia \
      --context-name "$LABEL" \
      --patients "$PATIENTS" \
      --items "$ITEMS" \
      --pool "$POOL" \
      --corpus-dir "$LUMIA_XML_DIR" \
      --output-dir "$OUT_DIR" \
      --query-split test \
      --tasks guo_readmission \
      --model "$MODEL" \
      --max-chars-per-patient "$CHARS" \
      --vertex-input-uri "$GCS_INPUT" \
      --vertex-output-prefix "$GCS_OUTPUT"
    echo ""
done

echo "--- [$(date +%H:%M:%S)] Analyzing results ---"
uv run python experiments/fewshot_with_labels/analyze_results.py \
  --input-dir "$OUT_DIR" \
  --output-dir "$OUT_DIR"

echo ""
echo "===== Done ====="
echo "  outputs: $OUT_DIR/experiment_results_baseline_lumia_*.jsonl"
