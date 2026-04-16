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

# Iteration 4: focus on best short-run candidates
COMMON_ARGS="--use-transformer --use-attention --detailed-eval --epochs 10 --batch-size 1 --scale 0.1 --classes 6 --num-workers 4 --prefetch-factor 2 --persistent-workers"

# A) Baseline reference
run_experiment "A_Baseline" \
    "python train.py ${COMMON_ARGS} --learning-rate 1e-4 --no-class-weights --no-rare-oversampling --results-dir results_ablation4_A_baseline"

# B) WeightsSoft (current best mIoU among short runs)
run_experiment "B_WeightsSoft_p050" \
    "python train.py ${COMMON_ARGS} --learning-rate 1e-4 --use-class-weights --class-weight-power 0.5 --class-weight-min 0.25 --class-weight-max 3.0 --no-rare-oversampling --results-dir results_ablation4_B_weights_soft_p050"

# C) WeightsSoft (milder weighting)
run_experiment "C_WeightsSoft_p040" \
    "python train.py ${COMMON_ARGS} --learning-rate 1e-4 --use-class-weights --class-weight-power 0.4 --class-weight-min 0.25 --class-weight-max 3.0 --no-rare-oversampling --results-dir results_ablation4_C_weights_soft_p040"

# D) WeightsSoft + very light oversampling
run_experiment "D_WeightsSoft_p040_Sampler020" \
    "python train.py ${COMMON_ARGS} --learning-rate 1e-4 --use-class-weights --class-weight-power 0.4 --class-weight-min 0.25 --class-weight-max 3.0 --use-rare-oversampling --oversampling-rarity-power 0.4 --oversampling-strength 0.2 --oversampling-max-sample-weight 1.5 --results-dir results_ablation4_D_weights_p040_sampler020"

GLOBAL_END_TS=$(date +%s)
TOTAL_DURATION=$((GLOBAL_END_TS - GLOBAL_START_TS))

echo
printf '%0.s=' {1..60}
echo
echo "ALL EXPERIMENTS COMPLETED"
printf 'Total elapsed time: %02d:%02d:%02d\n' $((TOTAL_DURATION / 3600)) $(((TOTAL_DURATION % 3600) / 60)) $((TOTAL_DURATION % 60))
printf '%0.s=' {1..60}
echo
