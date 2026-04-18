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

# Ablation 4 (6 classes): narrow search around ablation3 best Dice (B_Hybrid_sampler_plus).
# Memory-safe setup only.
COMMON_ARGS="--use-transformer --use-attention --detailed-eval --epochs 20 --batch-size 1 --scale 0.1 --classes 6 --num-workers 4 --prefetch-factor 2 --persistent-workers"

# A) Reference best from ablation 3
run_experiment "A_Hybrid_sampler_ref" \
    "python train.py ${COMMON_ARGS} --learning-rate 5e-5 --use-class-weights --class-weight-power 0.25 --class-weight-min 0.7 --class-weight-max 1.6 --use-rare-oversampling --oversampling-rarity-power 0.35 --oversampling-strength 0.15 --oversampling-max-sample-weight 1.6 --results-dir results_ablation4_A_hybrid_sampler_ref"

# B) Slightly weaker sampler strength (stability check)
run_experiment "B_Hybrid_sampler_s012" \
    "python train.py ${COMMON_ARGS} --learning-rate 5e-5 --use-class-weights --class-weight-power 0.25 --class-weight-min 0.7 --class-weight-max 1.6 --use-rare-oversampling --oversampling-rarity-power 0.35 --oversampling-strength 0.12 --oversampling-max-sample-weight 1.6 --results-dir results_ablation4_B_hybrid_sampler_s012"

# C) Slightly stronger sampler strength (rare-class push)
run_experiment "C_Hybrid_sampler_s018" \
    "python train.py ${COMMON_ARGS} --learning-rate 5e-5 --use-class-weights --class-weight-power 0.25 --class-weight-min 0.7 --class-weight-max 1.6 --use-rare-oversampling --oversampling-rarity-power 0.35 --oversampling-strength 0.18 --oversampling-max-sample-weight 1.6 --results-dir results_ablation4_C_hybrid_sampler_s018"

# D) Same as reference with tighter max sample weight
run_experiment "D_Hybrid_sampler_cap15" \
    "python train.py ${COMMON_ARGS} --learning-rate 5e-5 --use-class-weights --class-weight-power 0.25 --class-weight-min 0.7 --class-weight-max 1.6 --use-rare-oversampling --oversampling-rarity-power 0.35 --oversampling-strength 0.15 --oversampling-max-sample-weight 1.5 --results-dir results_ablation4_D_hybrid_sampler_cap15"

GLOBAL_END_TS=$(date +%s)
TOTAL_DURATION=$((GLOBAL_END_TS - GLOBAL_START_TS))

echo
printf '%0.s=' {1..60}
echo
echo "ALL EXPERIMENTS COMPLETED"
printf 'Total elapsed time: %02d:%02d:%02d\n' $((TOTAL_DURATION / 3600)) $(((TOTAL_DURATION % 3600) / 60)) $((TOTAL_DURATION % 60))
printf '%0.s=' {1..60}
echo
