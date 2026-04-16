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

# Iteration 7: verify class-6 improvement strategies with low risk to global Dice.
COMMON_ARGS="--use-transformer --use-attention --detailed-eval --epochs 20 --batch-size 1 --scale 0.1 --classes 6 --num-workers 4 --prefetch-factor 2 --persistent-workers"

# A) Reference baseline
run_experiment "A_Baseline_ref" \
    "python train.py ${COMMON_ARGS} --learning-rate 1e-4 --no-class-weights --no-rare-oversampling --results-dir results_ablation7_A_baseline_ref"

# B) Rare-class sampler (stronger than ultralight, still bounded)
run_experiment "B_Sampler_balanced" \
    "python train.py ${COMMON_ARGS} --learning-rate 1e-4 --no-class-weights --use-rare-oversampling --oversampling-rarity-power 0.45 --oversampling-strength 0.2 --oversampling-max-sample-weight 1.8 --results-dir results_ablation7_B_sampler_balanced"

# C) Hybrid mild weights + light sampler
run_experiment "C_Hybrid_mild" \
    "python train.py ${COMMON_ARGS} --learning-rate 1e-4 --use-class-weights --class-weight-power 0.3 --class-weight-min 0.6 --class-weight-max 1.8 --use-rare-oversampling --oversampling-rarity-power 0.35 --oversampling-strength 0.1 --oversampling-max-sample-weight 1.5 --results-dir results_ablation7_C_hybrid_mild"

GLOBAL_END_TS=$(date +%s)
TOTAL_DURATION=$((GLOBAL_END_TS - GLOBAL_START_TS))

echo
printf '%0.s=' {1..60}
echo
echo "ALL EXPERIMENTS COMPLETED"
printf 'Total elapsed time: %02d:%02d:%02d\n' $((TOTAL_DURATION / 3600)) $(((TOTAL_DURATION % 3600) / 60)) $((TOTAL_DURATION % 60))
printf '%0.s=' {1..60}
echo
