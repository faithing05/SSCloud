# A Baseline reference
INFO: Starting training:
        Epochs:          10
        Batch size:      1
        Learning rate:   0.0001
        Training size:   126
        Validation size: 13
        Checkpoints:     True
        Device:          cuda
        Images scaling:  0.1
        Mixed Precision: False
        Num workers:     4
        Persistent wrk:  True

etailed metrics saved to results_ablation4_A_baseline
  - Confusion matrix: results_ablation4_A_baseline/confusion_matrix_epoch10.png
  - Per-class IoU: results_ablation4_A_baseline/per_class_iou_epoch10.txt
  - Average inference time per image: 0.0096 seconds
INFO: Final Epoch 10 - Detailed Evaluation:
INFO:   Dice Score: 0.5649
INFO:   Average Inference Time per image: 0.0096 seconds
INFO:   Mean IoU (excluding NaN): 0.2521
INFO:   Per-Class IoU (first 10 classes):
INFO:     Class   0: IoU = 0.0319
INFO:     Class   1: IoU = 0.9037
INFO:     Class   2: IoU = 0.5020
INFO:     Class   3: IoU = 0.4806
INFO:     Class   4: IoU = 0.3308
INFO:     Class   5: IoU = 0.0000
INFO:     Class   6: IoU = 0.0200
INFO:     Class   7: IoU = 0.0000
INFO:     Class   8: IoU = 0.0000
INFO: Validation Dice score: 0.564887523651123
INFO: Checkpoint 10 saved to results_ablation4_A_baseline/checkpoints/checkpoint_epoch10.pth!
wandb: Waiting for W&B process to finish... (success).
wandb:
wandb: Run history:
wandb:           epoch ▁▁▁▁▂▂▂▂▃▃▃▃▃▃▃▃▄▄▄▄▅▅▅▅▆▆▆▆▆▆▆▆▇▇▇▇████
wandb:   learning rate ▁▁▁▁▁▁▁▁▁▁
wandb:            step ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:      train loss ▄▄▃▄▄▄▂▄▄▇▄▄▄▂▇▃▄█▃▁▄▇▇▁▁▁▄▃▁▂▂▄▂▂▅▂▄▆▄▄
wandb: validation Dice ▁█▄▄▅▄█▄▅▅
wandb:
wandb: Run summary:
wandb:           epoch 10
wandb:   learning rate 0.0001
wandb:            step 1260
wandb:      train loss 0.57693
wandb: validation Dice 0.56489
wandb:
wandb: You can sync this run to the cloud by running:
wandb: wandb sync /workspace/SSCloud/train/wandb/offline-run-20260413_185611-sdp9wz2u
wandb: Find logs at: ./wandb/offline-run-20260413_185611-sdp9wz2u/logs

DONE: [A_Baseline]

Stage duration: 00:07:39

# B WeightsSoft (current best mIoU among short runs)
INFO: Starting training:
        Epochs:          10
        Batch size:      1
        Learning rate:   0.0001
        Training size:   126
        Validation size: 13
        Checkpoints:     True
        Device:          cuda
        Images scaling:  0.1
        Mixed Precision: False
        Num workers:     4
        Persistent wrk:  True

Detailed metrics saved to results_ablation4_B_weights_soft_p050
  - Confusion matrix: results_ablation4_B_weights_soft_p050/confusion_matrix_epoch10.png
  - Per-class IoU: results_ablation4_B_weights_soft_p050/per_class_iou_epoch10.txt
  - Average inference time per image: 0.0081 seconds
INFO: Final Epoch 10 - Detailed Evaluation:
INFO:   Dice Score: 0.5608
INFO:   Average Inference Time per image: 0.0081 seconds
INFO:   Mean IoU (excluding NaN): 0.2827
INFO:   Per-Class IoU (first 10 classes):
INFO:     Class   0: IoU = 0.1566
INFO:     Class   1: IoU = 0.9065
INFO:     Class   2: IoU = 0.5552
INFO:     Class   3: IoU = 0.4964
INFO:     Class   4: IoU = 0.3353
INFO:     Class   5: IoU = 0.0000
INFO:     Class   6: IoU = 0.0947
INFO:     Class   7: IoU = 0.0000
INFO:     Class   8: IoU = 0.0000
INFO: Validation Dice score: 0.5608178377151489
INFO: Checkpoint 10 saved to results_ablation4_B_weights_soft_p050/checkpoints/checkpoint_epoch10.pth!
wandb: Waiting for W&B process to finish... (success).
wandb:
wandb: Run history:
wandb:           epoch ▁▁▁▁▂▂▂▂▃▃▃▃▃▃▃▃▄▄▄▄▅▅▅▅▆▆▆▆▆▆▆▆▇▇▇▇████
wandb:   learning rate █████████▁
wandb:            step ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:      train loss ▆█▆▅▃▅▇█▆▇▃█▄▄▅▆▅▃▂▆▄▇▆▂▃▅▁▃▆▅▃▄▄▄▂▁▄▃▂▅
wandb: validation Dice ▁▅▄█▅▅▆▂▆▆
wandb:
wandb: Run summary:
wandb:           epoch 10
wandb:   learning rate 1e-05
wandb:            step 1260
wandb:      train loss 1.92017
wandb: validation Dice 0.56082
wandb:
wandb: You can sync this run to the cloud by running:
wandb: wandb sync /workspace/SSCloud/train/wandb/offline-run-20260413_190350-zlulmf1f
wandb: Find logs at: ./wandb/offline-run-20260413_190350-zlulmf1f/logs

DONE: [B_WeightsSoft_p050]

Stage duration: 00:07:32

# C WeightsSoft (milder weighting)
INFO: Starting training:
        Epochs:          10
        Batch size:      1
        Learning rate:   0.0001
        Training size:   126
        Validation size: 13
        Checkpoints:     True
        Device:          cuda
        Images scaling:  0.1
        Mixed Precision: False
        Num workers:     4
        Persistent wrk:  True

Detailed metrics saved to results_ablation4_C_weights_soft_p040
  - Confusion matrix: results_ablation4_C_weights_soft_p040/confusion_matrix_epoch10.png
  - Per-class IoU: results_ablation4_C_weights_soft_p040/per_class_iou_epoch10.txt
  - Average inference time per image: 0.0081 seconds
INFO: Final Epoch 10 - Detailed Evaluation:
INFO:   Dice Score: 0.5552
INFO:   Average Inference Time per image: 0.0081 seconds
INFO:   Mean IoU (excluding NaN): 0.2672
INFO:   Per-Class IoU (first 10 classes):
INFO:     Class   0: IoU = 0.0644
INFO:     Class   1: IoU = 0.9032
INFO:     Class   2: IoU = 0.4867
INFO:     Class   3: IoU = 0.5174
INFO:     Class   4: IoU = 0.3311
INFO:     Class   5: IoU = 0.0000
INFO:     Class   6: IoU = 0.1019
INFO:     Class   7: IoU = 0.0000
INFO:     Class   8: IoU = 0.0000
INFO: Validation Dice score: 0.555181086063385
INFO: Checkpoint 10 saved to results_ablation4_C_weights_soft_p040/checkpoints/checkpoint_epoch10.pth!
wandb: Waiting for W&B process to finish... (success).
wandb:
wandb: Run history:
wandb:           epoch ▁▁▁▁▂▂▂▂▃▃▃▃▃▃▃▃▄▄▄▄▅▅▅▅▆▆▆▆▆▆▆▆▇▇▇▇████
wandb:   learning rate ▁▁▁▁▁▁▁▁▁▁
wandb:            step ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:      train loss ▃▂▆▂▄▂▂▄▂▃▄▃▄▁▃▂▅▂▁▂▂▂▃▃▃▂█▃▂▄▂▁▃▄▁▁▄▁▄▁
wandb: validation Dice ▂▆▇█▁▇▃▆█▇
wandb:
wandb: Run summary:
wandb:           epoch 10
wandb:   learning rate 0.0001
wandb:            step 1260
wandb:      train loss 2.56641
wandb: validation Dice 0.55518
wandb:
wandb: You can sync this run to the cloud by running:
wandb: wandb sync /workspace/SSCloud/train/wandb/offline-run-20260413_191122-pqyc9qir
wandb: Find logs at: ./wandb/offline-run-20260413_191122-pqyc9qir/logs

DONE: [C_WeightsSoft_p040]

Stage duration: 00:07:34

# D WeightsSoft + very light oversampling
INFO: Starting training:
        Epochs:          10
        Batch size:      1
        Learning rate:   0.0001
        Training size:   126
        Validation size: 13
        Checkpoints:     True
        Device:          cuda
        Images scaling:  0.1
        Mixed Precision: False
        Num workers:     4
        Persistent wrk:  True

Detailed metrics saved to results_ablation4_D_weights_p040_sampler020
  - Confusion matrix: results_ablation4_D_weights_p040_sampler020/confusion_matrix_epoch10.png
  - Per-class IoU: results_ablation4_D_weights_p040_sampler020/per_class_iou_epoch10.txt
  - Average inference time per image: 0.0107 seconds
INFO: Final Epoch 10 - Detailed Evaluation:
INFO:   Dice Score: 0.4609
INFO:   Average Inference Time per image: 0.0107 seconds
INFO:   Mean IoU (excluding NaN): 0.2478
INFO:   Per-Class IoU (first 10 classes):
INFO:     Class   0: IoU = 0.0989
INFO:     Class   1: IoU = 0.9080
INFO:     Class   2: IoU = 0.3885
INFO:     Class   3: IoU = 0.5015
INFO:     Class   4: IoU = 0.3092
INFO:     Class   5: IoU = 0.0000
INFO:     Class   6: IoU = 0.0231
INFO:     Class   7: IoU = 0.0000
INFO:     Class   8: IoU = 0.0008
INFO: Validation Dice score: 0.46089309453964233
INFO: Checkpoint 10 saved to results_ablation4_D_weights_p040_sampler020/checkpoints/checkpoint_epoch10.pth!
wandb: Waiting for W&B process to finish... (success).
wandb:
wandb: Run history:
wandb:           epoch ▁▁▁▁▂▂▂▂▃▃▃▃▃▃▃▃▄▄▄▄▅▅▅▅▆▆▆▆▆▆▆▆▇▇▇▇████
wandb:   learning rate █████████▁
wandb:            step ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:      train loss ▃▂▃▃█▃▇█▄▁▃▅▅▂▂▂▁▂▂▂▃▂▁▂▂▂▁▃▂▄▃▄▂▂▄▃▄▄▂▁
wandb: validation Dice ▆▁▆█▇▅▅▆▅▆
wandb:
wandb: Run summary:
wandb:           epoch 10
wandb:   learning rate 1e-05
wandb:            step 1260
wandb:      train loss 1.11057
wandb: validation Dice 0.46089
wandb:
wandb: You can sync this run to the cloud by running:
wandb: wandb sync /workspace/SSCloud/train/wandb/offline-run-20260413_191857-taie0j5b
wandb: Find logs at: ./wandb/offline-run-20260413_191857-taie0j5b/logs

DONE: [D_WeightsSoft_p040_Sampler020]
End time: 19:24:30
Stage duration: 00:07:15

============================================================
ALL EXPERIMENTS COMPLETED
Total elapsed time: 00:30:00
============================================================

Отличный прогон, картина стала очень понятной.
- По Dice лучший снова A_Baseline: 0.5649 (summary_run_ablation_4.md:20).
- По mIoU лучший B_WeightsSoft_p050: 0.2827 (summary_run_ablation_4.md:80), при почти таком же Dice 0.5608.
- C_WeightsSoft_p040 хуже B по обоим ключевым метрикам (Dice 0.5552, mIoU 0.2672).
- D_WeightsSoft_p040_Sampler020 заметно провалился (Dice 0.4609, mIoU 0.2478) — даже очень мягкий sampler здесь скорее вредит.
- Скорость: B/C самые быстрые (0.0081 s/img), A чуть медленнее (0.0096), D снова самый медленный (0.0107).
Итог по выбору:
- Если приоритет — максимальный Dice: брать A_Baseline.
- Если нужен компромисс Dice + лучший mIoU/редкие классы: брать B_WeightsSoft_p050.
- Ветка с oversampling на текущих настройках — нецелесообразна.
Что логично делать дальше:
1. На full (50 эпох) оставить только A_Baseline и B_WeightsSoft_p050.  
2. Если ресурсов хватает — добавить 1 контрольный запуск B с --class-weight-power 0.45 как промежуточный вариант.