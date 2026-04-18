# A Best Dice candidate from ablation 3
INFO: Starting training:
        Epochs:          50
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

Detailed metrics saved to results_ablation5_A_hybrid_sampler_plus_full
  - Confusion matrix: results_ablation5_A_hybrid_sampler_plus_full/confusion_matrix_epoch50.png
  - Per-class IoU: results_ablation5_A_hybrid_sampler_plus_full/per_class_iou_epoch50.txt
  - Average inference time per image: 0.0089 seconds
INFO: Final Epoch 50 - Detailed Evaluation:
INFO:   Dice Score: 0.5387
INFO:   Average Inference Time per image: 0.0089 seconds
INFO:   Mean IoU (excluding NaN): 0.3801
INFO:   Per-Class IoU (first 10 classes):
INFO:     Class   0: IoU = 0.7444
INFO:     Class   1: IoU = 0.5348
INFO:     Class   2: IoU = 0.6038
INFO:     Class   3: IoU = 0.3977
INFO:     Class   4: IoU = 0.0000
INFO:     Class   5: IoU = 0.0000
INFO: Validation Dice score: 0.5387182235717773
INFO: Checkpoint 50 saved to results_ablation5_A_hybrid_sampler_plus_full/checkpoints/checkpoint_epoch50.pth!
wandb: Waiting for W&B process to finish... (success).
wandb:
wandb: Run history:
wandb:           epoch ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:   learning rate ██████▂▂▂▂▂▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:            step ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇███
wandb:      train loss █▃▃▁▂▃▃▂▃▇▄▂▂▂▃▂▃▂▂▂▃▂▂▂▃▂▃▁▃▂▁▁▂▂▁▂▁▂▃▂
wandb: validation Dice ▄▇▃▂▅▁▅▆▇▆▅▇▄▅▇▅▆█▆▆███▇█▇█▆█▃▇▆▄▆▆▆▆▅▆█
wandb:
wandb: Run summary:
wandb:           epoch 50
wandb:   learning rate 0.0
wandb:            step 5350
wandb:      train loss 1.32666
wandb: validation Dice 0.53872
wandb:
wandb: You can sync this run to the cloud by running:
wandb: wandb sync /workspace/SSCloud/train/wandb/offline-run-20260418_160319-exmfifak
wandb: Find logs at: ./wandb/offline-run-20260418_160319-exmfifak/logs

DONE: [A_Hybrid_sampler_plus_full]

Stage duration: 00:23:01

# B Best mIoU/balance candidate from ablation 4
INFO: Starting training:
        Epochs:          50
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

Detailed metrics saved to results_ablation5_B_hybrid_sampler_cap15_full
  - Confusion matrix: results_ablation5_B_hybrid_sampler_cap15_full/confusion_matrix_epoch50.png
  - Per-class IoU: results_ablation5_B_hybrid_sampler_cap15_full/per_class_iou_epoch50.txt
  - Average inference time per image: 0.0094 seconds
INFO: Final Epoch 50 - Detailed Evaluation:
INFO:   Dice Score: 0.4371
INFO:   Average Inference Time per image: 0.0094 seconds
INFO:   Mean IoU (excluding NaN): 0.3868
INFO:   Per-Class IoU (first 10 classes):
INFO:     Class   0: IoU = 0.7470
INFO:     Class   1: IoU = 0.5521
INFO:     Class   2: IoU = 0.6039
INFO:     Class   3: IoU = 0.4179
INFO:     Class   4: IoU = 0.0000
INFO:     Class   5: IoU = 0.0000
INFO: Validation Dice score: 0.4370500445365906
INFO: Checkpoint 50 saved to results_ablation5_B_hybrid_sampler_cap15_full/checkpoints/checkpoint_epoch50.pth!
wandb: Waiting for W&B process to finish... (success).
wandb:
wandb: Run history:
wandb:           epoch ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:   learning rate ████████▂▂▂▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:            step ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇███
wandb:      train loss ▄▄▃▅▃▃█▂▂▂▂▂▄▃▅▃▆▂▂▃▂▄▂▃▁▂▆▅▃▃▂▅▂▂▅▃▃▂▃▂
wandb: validation Dice ▃▅▁▇█▄▅▅▃▅▅▅▅▅▅▆▅▇▇▇▇▇▇▅▅▇▆▇▆▇▅▆▇▇▇▆▆▆▅▆
wandb:
wandb: Run summary:
wandb:           epoch 50
wandb:   learning rate 0.0
wandb:            step 5350
wandb:      train loss 0.60722
wandb: validation Dice 0.43705
wandb:
wandb: You can sync this run to the cloud by running:
wandb: wandb sync /workspace/SSCloud/train/wandb/offline-run-20260418_162618-9rvlj91g
wandb: Find logs at: ./wandb/offline-run-20260418_162618-9rvlj91g/logs

DONE: [B_Hybrid_sampler_cap15_full]

Stage duration: 00:22:37

============================================================
ALL EXPERIMENTS COMPLETED
Total elapsed time: 00:45:38
============================================================

Разобрал train/ablations/ablation_with_6_classes/summary/summary_run_ablation_5.md.
Итог ablation 5:
- A_Hybrid_sampler_plus_full
  - Dice: 0.5387
  - Mean IoU: 0.3801
  - Inference: 0.0089 s/img
  - Checkpoint: results_ablation5_A_hybrid_sampler_plus_full/checkpoints/checkpoint_epoch50.pth
- B_Hybrid_sampler_cap15_full
  - Dice: 0.4371
  - Mean IoU: 0.3868
  - Inference: 0.0094 s/img
  - Checkpoint: results_ablation5_B_hybrid_sampler_cap15_full/checkpoints/checkpoint_epoch50.pth
Вывод:
- Если главный KPI — Dice, однозначно выбираем A (разница очень большая: +0.1016).
- B чуть лучше по mIoU, но просадка по Dice слишком сильная.
- По редким классам прогресса нет у обоих: class 4 = 0, class 5 = 0.
Решение для трека 6 классов:
- Финальный кандидат: A_Hybrid_sampler_plus_full.
- Далее делай inference по его checkpoint и фиксируй как финальный результат трека.
Если нужно, следующим сообщением дам готовую команду inference именно под этот checkpoint (bash и PowerShell варианты).