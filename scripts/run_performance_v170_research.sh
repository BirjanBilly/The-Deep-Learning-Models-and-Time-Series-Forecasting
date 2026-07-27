#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/financial_risk_graph_v170.yaml}"
FOLDS="${FOLDS:-development}"
SEEDS="${SEEDS:-11 22 33}"
TRAIN_DEVICE="${TRAIN_DEVICE:-auto}"
EVAL_DEVICE="${EVAL_DEVICE:-auto}"
FORCE_REEVALUATE="${FORCE_REEVALUATE:-0}"

if [[ "$FOLDS" != "development" ]]; then
    echo "Formal v1.7 development runner accepts only FOLDS=development." >&2
    echo "The reserved 2025 fold must remain untouched." >&2
    exit 2
fi

mkdir -p logs

run_stage() {
    local name="$1"
    shift
    echo
    echo "============================================================"
    echo "Starting: ${name}"
    date --iso-8601=seconds
    echo "Command: $*"
    echo "============================================================"
    "$@" 2>&1 | tee "logs/${name}.log"
    echo "Completed: ${name}"
    date --iso-8601=seconds
}

if [[ ! -f data/processed/market_panel_v150_core.npz || ! -f data/processed/market_panel_v150_long.npz ]]; then
    echo "Missing processed panels. Run the audited v1.5 data download/build workflow first." >&2
    exit 2
fi

read -r -a SEED_ARGS <<< "$SEEDS"

run_stage performance_v170_build_statistical \
    python scripts/build_statistical_experts_v170.py \
        --config "$CONFIG" --folds "$FOLDS" --skip-existing

run_stage performance_v170_pretrain_structured \
    python scripts/pretrain_structured_v170.py \
        --config "$CONFIG" --folds "$FOLDS" --seeds "${SEED_ARGS[@]}" \
        --device "$TRAIN_DEVICE" --skip-existing

run_stage performance_v170_train_structured \
    python scripts/train_structured_v170.py \
        --config "$CONFIG" --folds "$FOLDS" --seeds "${SEED_ARGS[@]}" \
        --device "$TRAIN_DEVICE" --skip-existing

run_stage performance_v170_export_structured \
    python scripts/export_structured_seed_v170.py \
        --config "$CONFIG" --folds "$FOLDS" --seeds "${SEED_ARGS[@]}" \
        --device "$EVAL_DEVICE" --skip-existing

run_stage performance_v170_train_tailgan \
    python scripts/train_tail_v170.py \
        --config "$CONFIG" --folds "$FOLDS" --seeds "${SEED_ARGS[@]}" \
        --objective tailgan --device "$TRAIN_DEVICE" --skip-existing

run_stage performance_v170_export_tailgan \
    python scripts/export_tail_seed_v170.py \
        --config "$CONFIG" --folds "$FOLDS" --seeds "${SEED_ARGS[@]}" \
        --objective tailgan --device "$EVAL_DEVICE" --skip-existing

run_stage performance_v170_train_gom \
    python scripts/train_tail_v170.py \
        --config "$CONFIG" --folds "$FOLDS" --seeds "${SEED_ARGS[@]}" \
        --objective gom --device "$TRAIN_DEVICE" --skip-existing

run_stage performance_v170_export_gom \
    python scripts/export_tail_seed_v170.py \
        --config "$CONFIG" --folds "$FOLDS" --seeds "${SEED_ARGS[@]}" \
        --objective gom --device "$EVAL_DEVICE" --skip-existing

ENSEMBLE_SKIP=()
if [[ "$FORCE_REEVALUATE" != "1" ]]; then
    ENSEMBLE_SKIP=(--skip-existing)
fi
run_stage performance_v170_evaluate_ensembles \
    python scripts/evaluate_ensembles_v170.py \
        --config "$CONFIG" --folds "$FOLDS" --seeds "${SEED_ARGS[@]}" \
        "${ENSEMBLE_SKIP[@]}"

run_stage performance_v170_compare \
    python scripts/compare_performance_v170.py --config "$CONFIG"
run_stage performance_v170_report \
    python scripts/make_performance_v170_report.py --config "$CONFIG"
run_stage performance_v170_verify \
    python scripts/verify_performance_v170.py --config "$CONFIG"

echo "PERFORMANCE V1.7 STATISTICAL-CHAMPION RESEARCH MATRIX COMPLETED SUCCESSFULLY"
