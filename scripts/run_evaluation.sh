#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────
# run_evaluation.sh
# Full ASR-RAG end-to-end evaluation on QASPER + SciDocs-RAG
# ─────────────────────────────────────────────────────────

set -e

CONFIG="configs/config.yaml"
RESULTS="results/"
MAX_QUERIES=500
WARMUP=100
RECORD_EVERY=200

echo "======================================================"
echo "  ASR-RAG Full Evaluation"
echo "  Config    : $CONFIG"
echo "  Max queries: $MAX_QUERIES per dataset"
echo "  Warm-up   : $WARMUP queries"
echo "======================================================"

# ── QASPER ────────────────────────────────────────────────
echo ""
echo ">>> Running QASPER experiment (all systems)..."
python experiments/run_qasper.py \
    --system       all \
    --config       $CONFIG \
    --max_queries  $MAX_QUERIES \
    --warmup       $WARMUP \
    --record_every $RECORD_EVERY \
    --results_dir  ${RESULTS}qasper

# ── SciDocs-RAG ───────────────────────────────────────────
echo ""
echo ">>> Running SciDocs-RAG experiment (all systems)..."
python experiments/run_scidocs.py \
    --system       all \
    --config       $CONFIG \
    --max_queries  $MAX_QUERIES \
    --warmup       $WARMUP \
    --record_every $RECORD_EVERY \
    --results_dir  ${RESULTS}scidocs

# ── Summary ───────────────────────────────────────────────
echo ""
echo "======================================================"
echo "  Evaluation complete!"
echo "  Results saved in: $RESULTS"
echo "======================================================"
