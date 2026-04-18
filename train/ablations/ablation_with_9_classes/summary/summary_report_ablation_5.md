# Ablation 5 Final Report

## Scope

Compared 2 full-training (50 epochs) runs from `run_ablation_5.sh`:

- `A_Baseline`
- `B_WeightsSoft_p050`

Source log: `train/summary_run_ablation_5.md`

## Final Metrics

| Experiment | Dice | Mean IoU | Avg inference time (s/img) |
|---|---:|---:|---:|
| A_Baseline | 0.5667 | 0.2881 | 0.0086 |
| B_WeightsSoft_p050 | 0.5602 | 0.2852 | 0.0110 |

## Per-class IoU (classes 0-8)

| Class | A_Baseline | B_WeightsSoft_p050 | Delta (B - A) |
|---:|---:|---:|---:|
| 0 | 0.1425 | 0.1451 | +0.0026 |
| 1 | 0.9092 | 0.9160 | +0.0068 |
| 2 | 0.5336 | 0.5449 | +0.0113 |
| 3 | 0.5454 | 0.5513 | +0.0059 |
| 4 | 0.3592 | 0.3536 | -0.0056 |
| 5 | 0.0000 | 0.0000 | +0.0000 |
| 6 | 0.1027 | 0.0562 | -0.0465 |
| 7 | 0.0000 | 0.0000 | +0.0000 |
| 8 | 0.0000 | 0.0000 | +0.0000 |

## Decision

Select **A_Baseline** as the final model for this dataset/configuration.

Rationale:

1. Best global quality: higher Dice and higher Mean IoU.
2. Better runtime: lower inference time per image.
3. More stable class balance in aggregate despite small gains of B on classes 1-3.

## Recommended checkpoint

`results_ablation5_A_baseline_full/checkpoints/checkpoint_epoch50.pth`

## Notes for next iteration

- If class-6 quality becomes critical, test a targeted class-6 strategy (localized weighting or class-specific augmentation) without enabling global oversampling.
- Keep baseline as the reference run for all future changes.
