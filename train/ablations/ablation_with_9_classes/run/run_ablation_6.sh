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

# Iteration 6: low-cost, high-signal check for rare classes.
# 20 epochs to reduce GPU time but still observe stable trends.
COMMON_ARGS="--use-transformer --use-attention --detailed-eval --epochs 20 --batch-size 1 --scale 0.1 --classes 9 --num-workers 4 --prefetch-factor 2 --persistent-workers"

# A) Control: proven baseline.
run_experiment "A_Baseline_ctrl" \
    "python train.py ${COMMON_ARGS} --learning-rate 1e-4 --no-class-weights --no-rare-oversampling --results-dir results_ablation6_A_baseline_ctrl"

# B) Very mild class-weights (test rare-class gain with minimal Dice risk).
run_experiment "B_Weights_mild" \
    "python train.py ${COMMON_ARGS} --learning-rate 1e-4 --use-class-weights --class-weight-power 0.35 --class-weight-min 0.5 --class-weight-max 2.0 --no-rare-oversampling --results-dir results_ablation6_B_weights_mild"

# C) Ultra-light sampler only (checks class-6 lift without strong distribution shift).
run_experiment "C_Sampler_ultralight" \
    "python train.py ${COMMON_ARGS} --learning-rate 1e-4 --no-class-weights --use-rare-oversampling --oversampling-rarity-power 0.3 --oversampling-strength 0.15 --oversampling-max-sample-weight 1.4 --results-dir results_ablation6_C_sampler_ultralight"

GLOBAL_END_TS=$(date +%s)
TOTAL_DURATION=$((GLOBAL_END_TS - GLOBAL_START_TS))

echo
printf '%0.s=' {1..60}
echo
echo "ALL EXPERIMENTS COMPLETED"
printf 'Total elapsed time: %02d:%02d:%02d\n' $((TOTAL_DURATION / 3600)) $(((TOTAL_DURATION % 3600) / 60)) $((TOTAL_DURATION % 60))
printf '%0.s=' {1..60}
echo
