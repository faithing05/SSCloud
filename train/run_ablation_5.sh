#!/usr/bin/env bash
set -euo pipefail

export WANDB_MODE=offline

GLOBAL_START_TS=$(date +%s)

run_experiment() {
    local name="$1"
    local cmd="$2"
    local start_ts end_ts duration

    start_ts=$(date +%s)
    echo
    printf '%0.s=' {1..60}
    echo
    echo "START EXPERIMENT: [${name}]"
    echo "Start time: $(date '+%H:%M:%S')"
    printf '%0.s-' {1..60}
    echo

    eval "$cmd"

    end_ts=$(date +%s)
    duration=$((end_ts - start_ts))
    printf '%0.s-' {1..60}
    echo
    echo "DONE: [${name}]"
    echo "End time: $(date '+%H:%M:%S')"
    printf 'Stage duration: %02d:%02d:%02d\n' $((duration / 3600)) $(((duration % 3600) / 60)) $((duration % 60))
}

# Iteration 5 (full): top-2 candidates from short ablation
COMMON_ARGS="--use-transformer --use-attention --detailed-eval --epochs 50 --batch-size 1 --scale 0.1 --classes 6 --num-workers 4 --prefetch-factor 2 --persistent-workers"

# A) Baseline reference
run_experiment "A_Baseline" \
    "python train.py ${COMMON_ARGS} --learning-rate 1e-4 --no-class-weights --no-rare-oversampling --results-dir results_ablation5_A_baseline_full"

# B) WeightsSoft p=0.50 (best mIoU with near-baseline Dice)
run_experiment "B_WeightsSoft_p050" \
    "python train.py ${COMMON_ARGS} --learning-rate 1e-4 --use-class-weights --class-weight-power 0.5 --class-weight-min 0.25 --class-weight-max 3.0 --no-rare-oversampling --results-dir results_ablation5_B_weights_soft_p050_full"

GLOBAL_END_TS=$(date +%s)
TOTAL_DURATION=$((GLOBAL_END_TS - GLOBAL_START_TS))

echo
printf '%0.s=' {1..60}
echo
echo "ALL EXPERIMENTS COMPLETED"
printf 'Total elapsed time: %02d:%02d:%02d\n' $((TOTAL_DURATION / 3600)) $(((TOTAL_DURATION % 3600) / 60)) $((TOTAL_DURATION % 60))
printf '%0.s=' {1..60}
echo
