# A Baseline reference
INFO: Starting training:
        Epochs:          50
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

Detailed metrics saved to results_ablation5_A_baseline_full
  - Confusion matrix: results_ablation5_A_baseline_full/confusion_matrix_epoch50.png
  - Per-class IoU: results_ablation5_A_baseline_full/per_class_iou_epoch50.txt
  - Average inference time per image: 0.0086 seconds

INFO: Final Epoch 50 - Detailed Evaluation:
INFO:   Dice Score: 0.5667
INFO:   Average Inference Time per image: 0.0086 seconds
INFO:   Mean IoU (excluding NaN): 0.2881
INFO:   Per-Class IoU (first 10 classes):
INFO:     Class   0: IoU = 0.1425
INFO:     Class   1: IoU = 0.9092
INFO:     Class   2: IoU = 0.5336
INFO:     Class   3: IoU = 0.5454
INFO:     Class   4: IoU = 0.3592
INFO:     Class   5: IoU = 0.0000
INFO:     Class   6: IoU = 0.1027
INFO:     Class   7: IoU = 0.0000
INFO:     Class   8: IoU = 0.0000
INFO: Validation Dice score: 0.5666528940200806
INFO: Checkpoint 50 saved to results_ablation5_A_baseline_full/checkpoints/checkpoint_epoch50.pth!
wandb: Waiting for W&B process to finish... (success).
wandb:
wandb: Run history:
wandb:           epoch ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:   learning rate ██████▂▂▂▂▂▂▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:            step ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇███
wandb:      train loss ▅█▇▄▇▄▄▂▃▁▃▃▃▃▂▅▃▃▁▃▅▂▁▃▁▃▃▃▃▄▂▂▁▄▃▃▁▂▅▆
wandb: validation Dice ▅▇▅▄▁▄▅▅▅█▅▅▅▅▆▆▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▆▅▅▆▅▅
wandb:
wandb: Run summary:
wandb:           epoch 50
wandb:   learning rate 0.0
wandb:            step 6300
wandb:      train loss 0.50068
wandb: validation Dice 0.56665
wandb:
wandb: You can sync this run to the cloud by running:
wandb: wandb sync /workspace/SSCloud/train/wandb/offline-run-20260413_192916-burya234
wandb: Find logs at: ./wandb/offline-run-20260413_192916-burya234/logs

DONE: [A_Baseline]

Stage duration: 00:31:15

# B WeightsSoft p=0.50 (best mIoU with near-baseline Dice)
INFO: Starting training:
        Epochs:          50
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

Detailed metrics saved to results_ablation5_B_weights_soft_p050_full
  - Confusion matrix: results_ablation5_B_weights_soft_p050_full/confusion_matrix_epoch50.png
  - Per-class IoU: results_ablation5_B_weights_soft_p050_full/per_class_iou_epoch50.txt
  - Average inference time per image: 0.0110 seconds
INFO: Final Epoch 50 - Detailed Evaluation:
INFO:   Dice Score: 0.5602
INFO:   Average Inference Time per image: 0.0110 seconds
INFO:   Mean IoU (excluding NaN): 0.2852
INFO:   Per-Class IoU (first 10 classes):
INFO:     Class   0: IoU = 0.1451
INFO:     Class   1: IoU = 0.9160
INFO:     Class   2: IoU = 0.5449
INFO:     Class   3: IoU = 0.5513
INFO:     Class   4: IoU = 0.3536
INFO:     Class   5: IoU = 0.0000
INFO:     Class   6: IoU = 0.0562
INFO:     Class   7: IoU = 0.0000
INFO:     Class   8: IoU = 0.0000
INFO: Validation Dice score: 0.5602259039878845
INFO: Checkpoint 50 saved to results_ablation5_B_weights_soft_p050_full/checkpoints/checkpoint_epoch50.pth!
wandb: Waiting for W&B process to finish... (success).
wandb:
wandb: Run history:
wandb:           epoch ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:   learning rate ███████████▂▂▂▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:            step ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇███
wandb:      train loss ▂▂▃▃▅▄█▂▂▄▃▂▁▃▃▂▂▂▃▁▃▂▃▂▃▂▂▅▂▂▂▂▃▄▁▁▂▂▁▂
wandb: validation Dice ▄▃▁▁▄▂█▄█▅▆▅▆▆▆▆▆▅▆▆▆▆▆▆▆▆▆▆▆▆▆▆▆▆▆▆▆▆▆▆
wandb:
wandb: Run summary:
wandb:           epoch 50
wandb:   learning rate 0.0
wandb:            step 6300
wandb:      train loss 1.99882
wandb: validation Dice 0.56023
wandb:
wandb: You can sync this run to the cloud by running:
wandb: wandb sync /workspace/SSCloud/train/wandb/offline-run-20260413_200015-5xq4pztb
wandb: Find logs at: ./wandb/offline-run-20260413_200015-5xq4pztb/logs

DONE: [B_WeightsSoft_p050]

Stage duration: 00:32:08

============================================================
ALL EXPERIMENTS COMPLETED
Total elapsed time: 01:03:23
============================================================

Отличные новости: по full-прогону победитель уже понятен.
- A_Baseline: Dice 0.5667, mIoU 0.2881, 0.0086 s/img (summary_run_ablation_5.md:21, summary_run_ablation_5.md:23, summary_run_ablation_5.md:18)
- B_WeightsSoft_p050: Dice 0.5602, mIoU 0.2852, 0.0110 s/img (summary_run_ablation_5.md:79, summary_run_ablation_5.md:81, summary_run_ablation_5.md:77)
Что это значит:
- На full (50 эпох) Baseline лучше по всем ключевым агрегированным метрикам: и Dice, и mIoU, и скорость.
- Разница по Dice ~+0.0064 в пользу A, по mIoU ~+0.0029 в пользу A.
- У B есть локальные плюсы по отдельным классам (например class 1/2/3 чуть выше), но есть и заметная просадка по class 6 (0.0562 vs 0.1027), что, вероятно, и съедает общий итог.
Практический вывод:
- Для прод/финального чекпойнта выбирай results_ablation5_A_baseline_full/checkpoints/checkpoint_epoch50.pth.
- Ветка с WeightsSoft_p050 на этом датасете не дала устойчивого преимущества на длинном обучении.
Если хочешь, следующим шагом соберу короткий final_report_ablation5.md (1 страница: таблица, вывод, какой checkpoint использовать и почему).