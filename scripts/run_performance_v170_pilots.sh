#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/financial_risk_graph_v170.yaml}"
FOLDS="${FOLDS:-recent_2024}"
SEEDS="${SEEDS:-11 22 33}"
TRAIN_DEVICE="${TRAIN_DEVICE:-auto}"
EVAL_DEVICE="${EVAL_DEVICE:-auto}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-4}"
STRUCTURED_EPOCHS="${STRUCTURED_EPOCHS:-8}"
TAIL_EPOCHS="${TAIL_EPOCHS:-8}"

if [[ "$FOLDS" == "final" || "$FOLDS" == "untouched_2025" ]]; then
    echo "The reserved 2025 fold is prohibited in the v1.7 development pilot." >&2
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
    echo "Missing processed v1.5/v1.7 panels. Build the v1.5 long and core panels first." >&2
    exit 2
fi

read -r -a SEED_ARGS <<< "$SEEDS"
STRUCTURED_VARIANT="llmtime_target_ssl__statistical_champion__safe_pilot"
TAILGAN_VARIANT="tailgan_stable_factor_scale__probabilistic_safe_pilot"
GOM_VARIANT="gom_stress_shrunk_factor_scale__probabilistic_safe_pilot"

run_stage performance_v170_pilot_statistical \
    python scripts/build_statistical_experts_v170.py \
        --config "$CONFIG" --folds "$FOLDS"

run_stage performance_v170_pilot_pretrain_structured \
    python scripts/pretrain_structured_v170.py \
        --config "$CONFIG" --folds "$FOLDS" --seeds "${SEED_ARGS[@]}" \
        --device "$TRAIN_DEVICE" --epochs "$PRETRAIN_EPOCHS" \
        --variant "$STRUCTURED_VARIANT" --skip-existing

run_stage performance_v170_pilot_train_structured \
    python scripts/train_structured_v170.py \
        --config "$CONFIG" --folds "$FOLDS" --seeds "${SEED_ARGS[@]}" \
        --device "$TRAIN_DEVICE" --epochs "$STRUCTURED_EPOCHS" \
        --variant "$STRUCTURED_VARIANT" --skip-existing

run_stage performance_v170_pilot_export_structured \
    python scripts/export_structured_seed_v170.py \
        --config "$CONFIG" --folds "$FOLDS" --seeds "${SEED_ARGS[@]}" \
        --device "$EVAL_DEVICE" --variant "$STRUCTURED_VARIANT" --skip-existing

run_stage performance_v170_pilot_train_tailgan \
    python scripts/train_tail_v170.py \
        --config "$CONFIG" --folds "$FOLDS" --seeds "${SEED_ARGS[@]}" \
        --objective tailgan --device "$TRAIN_DEVICE" --epochs "$TAIL_EPOCHS" \
        --variant "$TAILGAN_VARIANT" --skip-existing

run_stage performance_v170_pilot_export_tailgan \
    python scripts/export_tail_seed_v170.py \
        --config "$CONFIG" --folds "$FOLDS" --seeds "${SEED_ARGS[@]}" \
        --objective tailgan --device "$EVAL_DEVICE" \
        --variant "$TAILGAN_VARIANT" --skip-existing

run_stage performance_v170_pilot_train_gom \
    python scripts/train_tail_v170.py \
        --config "$CONFIG" --folds "$FOLDS" --seeds "${SEED_ARGS[@]}" \
        --objective gom --device "$TRAIN_DEVICE" --epochs "$TAIL_EPOCHS" \
        --variant "$GOM_VARIANT" --skip-existing

run_stage performance_v170_pilot_export_gom \
    python scripts/export_tail_seed_v170.py \
        --config "$CONFIG" --folds "$FOLDS" --seeds "${SEED_ARGS[@]}" \
        --objective gom --device "$EVAL_DEVICE" \
        --variant "$GOM_VARIANT" --skip-existing

run_stage performance_v170_pilot_ensembles \
    python scripts/evaluate_ensembles_v170.py \
        --config "$CONFIG" --folds "$FOLDS" --seeds "${SEED_ARGS[@]}" \
        --structured-variant "$STRUCTURED_VARIANT" \
        --tailgan-variant "$TAILGAN_VARIANT" \
        --gom-variant "$GOM_VARIANT"

echo "PERFORMANCE V1.7 PILOT SUITE COMPLETED SUCCESSFULLY"
