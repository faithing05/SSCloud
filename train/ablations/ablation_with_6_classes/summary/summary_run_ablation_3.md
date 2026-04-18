# A Reference: best from ablation 2
INFO: Starting training:
        Epochs:          20
        Batch size:      1
        Learning rate:   5e-05
        Training size:   107
        Validation size: 11
        Checkpoints:     True
        Device:          cuda
        Images scaling:  0.1
        Mixed Precision: False
        Num workers:     4
        Persistent wrk:  True

Detailed metrics saved to results_ablation3_A_hybrid_ref
  - Confusion matrix: results_ablation3_A_hybrid_ref/confusion_matrix_epoch20.png
  - Per-class IoU: results_ablation3_A_hybrid_ref/per_class_iou_epoch20.txt
  - Average inference time per image: 0.0090 seconds
INFO: Final Epoch 20 - Detailed Evaluation:
INFO:   Dice Score: 0.3546
INFO:   Average Inference Time per image: 0.0090 seconds
INFO:   Mean IoU (excluding NaN): 0.3310
INFO:   Per-Class IoU (first 10 classes):
INFO:     Class   0: IoU = 0.7541
INFO:     Class   1: IoU = 0.2816
INFO:     Class   2: IoU = 0.6210
INFO:     Class   3: IoU = 0.3292
INFO:     Class   4: IoU = 0.0002
INFO:     Class   5: IoU = 0.0000
INFO: Validation Dice score: 0.35458317399024963
INFO: Checkpoint 20 saved to results_ablation3_A_hybrid_ref/checkpoints/checkpoint_epoch20.pth!
wandb: Waiting for W&B process to finish... (success).
wandb:
wandb: Run history:
wandb:           epoch ▁▁▁▁▂▂▂▂▂▂▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▇▇▇▇▇▇████
wandb:   learning rate ███████████▂▂▂▂▂▂▁▁▁
wandb:            step ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:      train loss ▂▅▂▂▅▁▃▅▃▂▃█▂▂▃▃▂▂▄▃▂▂▂▂▄▂▃▃▂▃▄▁▂▄▆▅▁▆▂▃
wandb: validation Dice ▆█▇▁██▄▃▇▆▃▁█▆▃▃▄▇▃▂
wandb:
wandb: Run summary:
wandb:           epoch 20
wandb:   learning rate 0.0
wandb:            step 2140
wandb:      train loss 0.51008
wandb: validation Dice 0.35458
wandb:
wandb: You can sync this run to the cloud by running:
wandb: wandb sync /workspace/SSCloud/train/wandb/offline-run-20260418_143625-tbuwqd3a
wandb: Find logs at: ./wandb/offline-run-20260418_143625-tbuwqd3a/logs

DONE: [A_Hybrid_ref]

Stage duration: 00:10:16

# B Slightly stronger sampler, same weights
INFO: Starting training:
        Epochs:          20
        Batch size:      1
        Learning rate:   5e-05
        Training size:   107
        Validation size: 11
        Checkpoints:     True
        Device:          cuda
        Images scaling:  0.1
        Mixed Precision: False
        Num workers:     4
        Persistent wrk:  True

Detailed metrics saved to results_ablation3_B_hybrid_sampler_plus
  - Confusion matrix: results_ablation3_B_hybrid_sampler_plus/confusion_matrix_epoch20.png
  - Per-class IoU: results_ablation3_B_hybrid_sampler_plus/per_class_iou_epoch20.txt
  - Average inference time per image: 0.0096 seconds
INFO: Final Epoch 20 - Detailed Evaluation:
INFO:   Dice Score: 0.5481
INFO:   Average Inference Time per image: 0.0096 seconds
INFO:   Mean IoU (excluding NaN): 0.3795
INFO:   Per-Class IoU (first 10 classes):
INFO:     Class   0: IoU = 0.7143
INFO:     Class   1: IoU = 0.5522
INFO:     Class   2: IoU = 0.5893
INFO:     Class   3: IoU = 0.4046
INFO:     Class   4: IoU = 0.0000
INFO:     Class   5: IoU = 0.0166
INFO: Validation Dice score: 0.548145055770874
INFO: Checkpoint 20 saved to results_ablation3_B_hybrid_sampler_plus/checkpoints/checkpoint_epoch20.pth!
wandb: Waiting for W&B process to finish... (success).
wandb:
wandb: Run history:
wandb:           epoch ▁▁▁▁▂▂▂▂▂▂▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▇▇▇▇▇▇████
wandb:   learning rate ████████████████▁▁▁▁
wandb:            step ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:      train loss ▇▅▃▁▄▅▂▃▃▂▅▄▆▂▂▁▂▅▃▃▃▁▂█▃▆▂▅▃▁▆▁▄▁▆▄▄▅▁█
wandb: validation Dice ▃▇▇▆▄▇▇█▇▁█▃▅▂▁▂▇▇▇█
wandb:
wandb: Run summary:
wandb:           epoch 20
wandb:   learning rate 1e-05
wandb:            step 2140
wandb:      train loss 1.79587
wandb: validation Dice 0.54815
wandb:
wandb: You can sync this run to the cloud by running:
wandb: wandb sync /workspace/SSCloud/train/wandb/offline-run-20260418_144642-nfrfslbo
wandb: Find logs at: ./wandb/offline-run-20260418_144642-nfrfslbo/logs

DONE: [B_Hybrid_sampler_plus]

Stage duration: 00:10:00

# C Milder weights + stronger sampler (shift balancing toward sampler)
INFO: Starting training:
        Epochs:          20
        Batch size:      1
        Learning rate:   5e-05
        Training size:   107
        Validation size: 11
        Checkpoints:     True
        Device:          cuda
        Images scaling:  0.1
        Mixed Precision: False
        Num workers:     4
        Persistent wrk:  True

Detailed metrics saved to results_ablation3_C_hybrid_sampler_focus
  - Confusion matrix: results_ablation3_C_hybrid_sampler_focus/confusion_matrix_epoch20.png
  - Per-class IoU: results_ablation3_C_hybrid_sampler_focus/per_class_iou_epoch20.txt
  - Average inference time per image: 0.0089 seconds
INFO: Final Epoch 20 - Detailed Evaluation:
INFO:   Dice Score: 0.3885
INFO:   Average Inference Time per image: 0.0089 seconds
INFO:   Mean IoU (excluding NaN): 0.3920
INFO:   Per-Class IoU (first 10 classes):
INFO:     Class   0: IoU = 0.7501
INFO:     Class   1: IoU = 0.5473
INFO:     Class   2: IoU = 0.6164
INFO:     Class   3: IoU = 0.4318
INFO:     Class   4: IoU = 0.0063
INFO:     Class   5: IoU = 0.0000
INFO: Validation Dice score: 0.3885403275489807
INFO: Checkpoint 20 saved to results_ablation3_C_hybrid_sampler_focus/checkpoints/checkpoint_epoch20.pth!
wandb: Waiting for W&B process to finish... (success).
wandb:
wandb: Run history:
wandb:           epoch ▁▁▁▁▂▂▂▂▂▂▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▇▇▇▇▇▇████
wandb:   learning rate ███████████▂▂▂▂▂▂▁▁▁
wandb:            step ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:      train loss ▄▃▃▃▄▄▂▃▃▆▂▃▄▂▅▂▃█▂▂▅▃▃▃▂▂▂▅▁▅▂▂▅▂▁▄▂▄▄▂
wandb: validation Dice ▅▆▆▃▂██▄▅▃▄▄▆▇▅▃▃▃▁▃
wandb:
wandb: Run summary:
wandb:           epoch 20
wandb:   learning rate 0.0
wandb:            step 2140
wandb:      train loss 1.42871
wandb: validation Dice 0.38854
wandb:
wandb: You can sync this run to the cloud by running:
wandb: wandb sync /workspace/SSCloud/train/wandb/offline-run-20260418_145640-rtvxjaaa
wandb: Find logs at: ./wandb/offline-run-20260418_145640-rtvxjaaa/logs

DONE: [C_Hybrid_sampler_focus]

Stage duration: 00:09:47

# D Weights focus + very light sampler (opposite direction)
INFO: Starting training:
        Epochs:          20
        Batch size:      1
        Learning rate:   5e-05
        Training size:   107
        Validation size: 11
        Checkpoints:     True
        Device:          cuda
        Images scaling:  0.1
        Mixed Precision: False
        Num workers:     4
        Persistent wrk:  True

Detailed metrics saved to results_ablation3_D_hybrid_weights_focus
  - Confusion matrix: results_ablation3_D_hybrid_weights_focus/confusion_matrix_epoch20.png
  - Per-class IoU: results_ablation3_D_hybrid_weights_focus/per_class_iou_epoch20.txt
  - Average inference time per image: 0.0082 seconds
INFO: Final Epoch 20 - Detailed Evaluation:
INFO:   Dice Score: 0.4280
INFO:   Average Inference Time per image: 0.0082 seconds
INFO:   Mean IoU (excluding NaN): 0.3920
INFO:   Per-Class IoU (first 10 classes):
INFO:     Class   0: IoU = 0.7350
INFO:     Class   1: IoU = 0.5543
INFO:     Class   2: IoU = 0.6165
INFO:     Class   3: IoU = 0.4347
INFO:     Class   4: IoU = 0.0030
INFO:     Class   5: IoU = 0.0088
INFO: Validation Dice score: 0.4280327558517456
INFO: Checkpoint 20 saved to results_ablation3_D_hybrid_weights_focus/checkpoints/checkpoint_epoch20.pth!
wandb: Waiting for W&B process to finish... (success).
wandb:
wandb: Run history:
wandb:           epoch ▁▁▁▁▂▂▂▂▂▂▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▇▇▇▇▇▇████
wandb:   learning rate █████████▂▂▂▂▂▂▁▁▁▁▁
wandb:            step ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:      train loss ▅▅▃▄▃▄▂▁▃▃▃▁▁▁▃▂▂▂█▇▃▂▂▂▃▃▃▂▃▃▂▂▃▄▃▂▃▃▃▁
wandb: validation Dice ▁▃▅█▃▅▂▃▃▄▄▅▅▄▃▃▃▃▄▅
wandb:
wandb: Run summary:
wandb:           epoch 20
wandb:   learning rate 0.0
wandb:            step 2140
wandb:      train loss 1.59773
wandb: validation Dice 0.42803
wandb:
wandb: You can sync this run to the cloud by running:
wandb: wandb sync /workspace/SSCloud/train/wandb/offline-run-20260418_150629-xmx9df35
wandb: Find logs at: ./wandb/offline-run-20260418_150629-xmx9df35/logs

DONE: [D_Hybrid_weights_focus]

Stage duration: 00:10:06

Разобрал train/ablations/ablation_with_6_classes/summary/summary_run_ablation_3.md.
Итоги по метрикам:
- A_Hybrid_ref: Dice 0.3546, mIoU 0.3310, time 0.0090
- B_Hybrid_sampler_plus: Dice 0.5481, mIoU 0.3795, time 0.0096
- C_Hybrid_sampler_focus: Dice 0.3885, mIoU 0.3920, time 0.0089
- D_Hybrid_weights_focus: Dice 0.4280, mIoU 0.3920, time 0.0082
Что это значит:
- По главной метрике Dice явный лидер — B (0.5481), и это лучший Dice среди всех запусков до сих пор в 6-классном треке.
- По mIoU лидируют C и D (0.3920), но у них заметно хуже Dice.
- Редкие классы:
  - У B лучший сигнал по class 5 (0.0166), но class 4 = 0.0000.
  - У C/D class 4/5 слегка “шевелятся”, но без качественного прорыва.
- A (референс) провалился аномально сильно относительно ablation 2 — лучше не использовать его как текущий ориентир.
Практический вывод:
- Брать B_Hybrid_sampler_plus как новый рабочий baseline для ablation 4.
- В ablation 4 делать очень узкий тюнинг вокруг B:
  - чуть вверх/вниз oversampling-strength (например 0.12, 0.15, 0.18)
  - один мягкий вариант max_sample_weight (1.5 vs 1.6)
  - веса оставить как в B (не усиливать), чтобы не уронить Dice.
Если хочешь, сразу напишу run_ablation_4.sh вокруг B в таком стиле.