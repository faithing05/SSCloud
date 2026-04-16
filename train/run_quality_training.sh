#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 [--results-dir DIR] [--epochs N] [--scale F] [--classes N] [--learning-rate LR] [--amp]"
    echo
    echo "High-quality baseline training preset (stable by our ablation results)."
    echo "Defaults: epochs=50, batch-size=1, scale=0.1, classes=9, lr=5e-5, AMP=off"
}

RESULTS_DIR="results_quality_baseline"
EPOCHS=50
BATCH_SIZE=1
SCALE=0.1
CLASSES=9
LEARNING_RATE=5e-5
USE_AMP=0
NUM_WORKERS=4
PREFETCH_FACTOR=2

while [[ $# -gt 0 ]]; do
    case "$1" in
        --results-dir)
            RESULTS_DIR="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --scale)
            SCALE="$2"
            shift 2
            ;;
        --classes)
            CLASSES="$2"
            shift 2
            ;;
        --learning-rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --amp)
            USE_AMP=1
            shift
            ;;
        --no-amp)
            USE_AMP=0
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            usage
            exit 1
            ;;
    esac
done

if [[ -z "${WANDB_MODE:-}" ]]; then
    export WANDB_MODE=offline
fi

START_TS=$(date +%s)
START_HUMAN=$(date '+%Y-%m-%d %H:%M:%S')

echo "Starting quality training preset..."
echo "  Start time:    ${START_HUMAN}"
echo "  Results dir:   ${RESULTS_DIR}"
echo "  Epochs:        ${EPOCHS}"
echo "  Batch size:    ${BATCH_SIZE}"
echo "  Scale:         ${SCALE}"
echo "  Classes:       ${CLASSES}"
echo "  Learning rate: ${LEARNING_RATE}"
echo "  AMP:           ${USE_AMP}"

cmd=(
    python train.py
    --use-transformer
    --use-attention
    --detailed-eval
    --epochs "${EPOCHS}"
    --batch-size "${BATCH_SIZE}"
    --scale "${SCALE}"
    --classes "${CLASSES}"
    --learning-rate "${LEARNING_RATE}"
    --no-class-weights
    --no-rare-oversampling
    --num-workers "${NUM_WORKERS}"
    --prefetch-factor "${PREFETCH_FACTOR}"
    --persistent-workers
    --results-dir "${RESULTS_DIR}"
)

if [[ "${USE_AMP}" -eq 1 ]]; then
    cmd+=(--amp)
fi

"${cmd[@]}"

END_TS=$(date +%s)
END_HUMAN=$(date '+%Y-%m-%d %H:%M:%S')
DURATION=$((END_TS - START_TS))

echo "Training finished. Best checkpoint is in: ${RESULTS_DIR}/checkpoints/"
echo "  End time:      ${END_HUMAN}"
printf '  Duration:      %02d:%02d:%02d\n' $((DURATION / 3600)) $(((DURATION % 3600) / 60)) $((DURATION % 60))
