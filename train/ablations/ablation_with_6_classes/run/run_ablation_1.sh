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

# Ablation 1 (6 classes):
# Baseline quality run achieved Dice=0.4186, mIoU=0.3853, while class 4/5 IoU is very low.
# Hypothesis: mild rebalancing (weights/sampler) and slightly larger input scale can lift rare classes
# without regressing global Dice too much.
COMMON_ARGS="--use-transformer --use-attention --detailed-eval --epochs 20 --batch-size 1 --classes 6 --num-workers 4 --prefetch-factor 2 --persistent-workers"

# A) Reference baseline (short-run control)
run_experiment "A_Baseline_ref_s010" \
    "python train.py ${COMMON_ARGS} --learning-rate 5e-5 --scale 0.1 --no-class-weights --no-rare-oversampling --results-dir results_ablation1_A_baseline_ref_s010"

# B) Mild class weights only
run_experiment "B_Weights_mild_s010" \
    "python train.py ${COMMON_ARGS} --learning-rate 5e-5 --scale 0.1 --use-class-weights --class-weight-power 0.35 --class-weight-min 0.5 --class-weight-max 2.0 --no-rare-oversampling --results-dir results_ablation1_B_weights_mild_s010"

# C) Ultra-light oversampling only
run_experiment "C_Sampler_ultralight_s010" \
    "python train.py ${COMMON_ARGS} --learning-rate 5e-5 --scale 0.1 --no-class-weights --use-rare-oversampling --oversampling-rarity-power 0.3 --oversampling-strength 0.15 --oversampling-max-sample-weight 1.4 --results-dir results_ablation1_C_sampler_ultralight_s010"

# D) Scale-up + mild weights
run_experiment "D_Weights_mild_s020" \
    "python train.py ${COMMON_ARGS} --learning-rate 5e-5 --scale 0.2 --use-class-weights --class-weight-power 0.3 --class-weight-min 0.6 --class-weight-max 1.8 --no-rare-oversampling --results-dir results_ablation1_D_weights_mild_s020"

GLOBAL_END_TS=$(date +%s)
TOTAL_DURATION=$((GLOBAL_END_TS - GLOBAL_START_TS))

echo
printf '%0.s=' {1..60}
echo
echo "ALL EXPERIMENTS COMPLETED"
printf 'Total elapsed time: %02d:%02d:%02d\n' $((TOTAL_DURATION / 3600)) $(((TOTAL_DURATION % 3600) / 60)) $((TOTAL_DURATION % 60))
printf '%0.s=' {1..60}
echo
