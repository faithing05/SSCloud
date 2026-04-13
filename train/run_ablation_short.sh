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

COMMON_ARGS="--use-transformer --use-attention --detailed-eval --epochs 10 --batch-size 1 --scale 0.1 --classes 9 --num-workers 4 --prefetch-factor 2 --persistent-workers"

# A) Baseline
run_experiment "A_Baseline" \
    "python train.py ${COMMON_ARGS} --learning-rate 1e-4 --no-class-weights --no-rare-oversampling --results-dir results_ablation_A_baseline_short"

# B) Smoothed class weights only
run_experiment "B_WeightsSoft" \
    "python train.py ${COMMON_ARGS} --learning-rate 1e-4 --use-class-weights --class-weight-power 0.5 --class-weight-min 0.25 --class-weight-max 3.0 --no-rare-oversampling --results-dir results_ablation_B_weights_soft_short"

# C) Soft oversampling only
run_experiment "C_SamplerSoft" \
    "python train.py ${COMMON_ARGS} --learning-rate 1e-4 --no-class-weights --use-rare-oversampling --oversampling-rarity-power 0.5 --oversampling-strength 0.35 --oversampling-max-sample-weight 2.0 --results-dir results_ablation_C_sampler_soft_short"

# D) Smoothed weights + soft oversampling
run_experiment "D_BothSoft" \
    "python train.py ${COMMON_ARGS} --learning-rate 1e-4 --use-class-weights --class-weight-power 0.5 --class-weight-min 0.25 --class-weight-max 3.0 --use-rare-oversampling --oversampling-rarity-power 0.5 --oversampling-strength 0.35 --oversampling-max-sample-weight 2.0 --results-dir results_ablation_D_both_soft_short"

GLOBAL_END_TS=$(date +%s)
TOTAL_DURATION=$((GLOBAL_END_TS - GLOBAL_START_TS))

echo
printf '%0.s=' {1..60}
echo
echo "ALL EXPERIMENTS COMPLETED"
printf 'Total elapsed time: %02d:%02d:%02d\n' $((TOTAL_DURATION / 3600)) $(((TOTAL_DURATION % 3600) / 60)) $((TOTAL_DURATION % 60))
printf '%0.s=' {1..60}
echo
