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

# Ablation 8 hypothesis:
# Rare classes have very low pixel share (esp. class IDs 64/67), so test larger image scale
# first, while avoiding aggressive weighting/sampling that previously hurt Dice.
COMMON_ARGS="--use-transformer --use-attention --detailed-eval --epochs 20 --batch-size 1 --classes 9 --num-workers 4 --prefetch-factor 2 --persistent-workers"

# A) Control baseline (current production reference)
run_experiment "A_Baseline_s010" \
    "python train.py ${COMMON_ARGS} --learning-rate 1e-4 --scale 0.1 --no-class-weights --no-rare-oversampling --results-dir results_ablation8_A_baseline_s010"

# B) Scale-up only (keep loss/sampler simple)
run_experiment "B_Baseline_s020" \
    "python train.py ${COMMON_ARGS} --learning-rate 1e-4 --scale 0.2 --no-class-weights --no-rare-oversampling --results-dir results_ablation8_B_baseline_s020"

# C) More aggressive scale-up to preserve tiny objects
run_experiment "C_Baseline_s025" \
    "python train.py ${COMMON_ARGS} --learning-rate 1e-4 --scale 0.25 --no-class-weights --no-rare-oversampling --results-dir results_ablation8_C_baseline_s025"

# D) Scale-up + very mild class weights (bounded to reduce Dice regression risk)
run_experiment "D_WeightsMild_s020" \
    "python train.py ${COMMON_ARGS} --learning-rate 1e-4 --scale 0.2 --use-class-weights --class-weight-power 0.25 --class-weight-min 0.7 --class-weight-max 1.6 --no-rare-oversampling --results-dir results_ablation8_D_weights_mild_s020"

GLOBAL_END_TS=$(date +%s)
TOTAL_DURATION=$((GLOBAL_END_TS - GLOBAL_START_TS))

echo
printf '%0.s=' {1..60}
echo
echo "ALL EXPERIMENTS COMPLETED"
printf 'Total elapsed time: %02d:%02d:%02d\n' $((TOTAL_DURATION / 3600)) $(((TOTAL_DURATION % 3600) / 60)) $((TOTAL_DURATION % 60))
printf '%0.s=' {1..60}
echo
