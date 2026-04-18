# A Baseline
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

Detailed metrics saved to results_ablation_A_baseline_full
  - Confusion matrix: results_ablation_A_baseline_full/confusion_matrix_epoch50.png
  - Per-class IoU: results_ablation_A_baseline_full/per_class_iou_epoch50.txt
  - Average inference time per image: 0.0109 seconds
INFO: Final Epoch 50 - Detailed Evaluation:
INFO:   Dice Score: 0.6450
INFO:   Average Inference Time per image: 0.0109 seconds
INFO:   Mean IoU (excluding NaN): 0.2921
INFO:   Per-Class IoU (first 10 classes):
INFO:     Class   0: IoU = 0.2630
INFO:     Class   1: IoU = 0.9194
INFO:     Class   2: IoU = 0.5335
INFO:     Class   3: IoU = 0.5562
INFO:     Class   4: IoU = 0.3572
INFO:     Class   5: IoU = 0.0000
INFO:     Class   6: IoU = 0.0000
INFO:     Class   7: IoU = 0.0000
INFO:     Class   8: IoU = 0.0000
INFO: Validation Dice score: 0.6449602842330933
INFO: Checkpoint 50 saved to results_ablation_A_baseline_full/checkpoints/checkpoint_epoch50.pth!
wandb: Waiting for W&B process to finish... (success).
wandb:
wandb: Run history:
wandb:           epoch ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:   learning rate ██████▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▁▁▁▁▁▁▁▁▁▁▁
wandb:            step ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇███
wandb:      train loss ▄▄▃▄▂▂▃▄▂▃▂█▂▃▇▂▂▃▂▃▃▁▂▁▂▃▂▁▁▃▁▃▂▁▃▄▁▁▁▃
wandb: validation Dice ▁▆▂▅▅▄▆▆▆▇▆▆▇▇▇▇▇▇▆▆█▆█▇█▇█▇▇▇▇█▇▇█▇████
wandb:
wandb: Run summary:
wandb:           epoch 50
wandb:   learning rate 0.0
wandb:            step 6300
wandb:      train loss 0.97497
wandb: validation Dice 0.64496
wandb:
wandb: You can sync this run to the cloud by running:
wandb: wandb sync /workspace/SSCloud/train/wandb/offline-run-20260413_160918-2zq2yn4d
wandb: Find logs at: ./wandb/offline-run-20260413_160918-2zq2yn4d/logs

DONE: [A_Baseline]

Stage duration: 00:31:06

# B Class weights only
INFO: Starting training:
        Epochs:          50
        Batch size:      1
        Learning rate:   8e-05
        Training size:   126
        Validation size: 13
        Checkpoints:     True
        Device:          cuda
        Images scaling:  0.1
        Mixed Precision: False
        Num workers:     4
        Persistent wrk:  True

Detailed metrics saved to results_ablation_B_weights_full
  - Confusion matrix: results_ablation_B_weights_full/confusion_matrix_epoch50.png
  - Per-class IoU: results_ablation_B_weights_full/per_class_iou_epoch50.txt
  - Average inference time per image: 0.0096 seconds
INFO: Final Epoch 50 - Detailed Evaluation:
INFO:   Dice Score: 0.4132
INFO:   Average Inference Time per image: 0.0096 seconds
INFO:   Mean IoU (excluding NaN): 0.2532
INFO:   Per-Class IoU (first 10 classes):
INFO:     Class   0: IoU = 0.1945
INFO:     Class   1: IoU = 0.7299
INFO:     Class   2: IoU = 0.5254
INFO:     Class   3: IoU = 0.4313
INFO:     Class   4: IoU = 0.3081
INFO:     Class   5: IoU = 0.0000
INFO:     Class   6: IoU = 0.0006
INFO:     Class   7: IoU = 0.0883
INFO:     Class   8: IoU = 0.0005
INFO: Validation Dice score: 0.4131702184677124
INFO: Checkpoint 50 saved to results_ablation_B_weights_full/checkpoints/checkpoint_epoch50.pth!
wandb: Waiting for W&B process to finish... (success).
wandb:
wandb: Run history:
wandb:           epoch ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:   learning rate ████████▂▂▂▂▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:            step ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇███
wandb:      train loss ▇▆▄▇█▄▇▆▃▃▃▂▄█▄▅▄▂▄▃▅▅▄█▄▂▄▃▃▂▄▄▆▁▃▄▄▂▃▄
wandb: validation Dice ▆▄▆▇▁▂▅▃▅█▇▇▂▂█▃▂▄▄▅▃▅▂▆▆▁▂▃▄▂█▆▇▅▂▂▂▄▇▃
wandb:
wandb: Run summary:
wandb:           epoch 50
wandb:   learning rate 0.0
wandb:            step 6300
wandb:      train loss 1.88905
wandb: validation Dice 0.41317
wandb:
wandb: You can sync this run to the cloud by running:
wandb: wandb sync /workspace/SSCloud/train/wandb/offline-run-20260413_164137-jwkwzvwp
wandb: Find logs at: ./wandb/offline-run-20260413_164137-jwkwzvwp/logs

DONE: [B_Weights]

Stage duration: 00:32:42

# C Oversampling only
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

Detailed metrics saved to results_ablation_C_sampler_full
  - Confusion matrix: results_ablation_C_sampler_full/confusion_matrix_epoch50.png
  - Per-class IoU: results_ablation_C_sampler_full/per_class_iou_epoch50.txt
  - Average inference time per image: 0.0095 seconds
INFO: Final Epoch 50 - Detailed Evaluation:
INFO:   Dice Score: 0.5619
INFO:   Average Inference Time per image: 0.0095 seconds
INFO:   Mean IoU (excluding NaN): 0.2670
INFO:   Per-Class IoU (first 10 classes):
INFO:     Class   0: IoU = 0.1264
INFO:     Class   1: IoU = 0.9153
INFO:     Class   2: IoU = 0.5079
INFO:     Class   3: IoU = 0.4858
INFO:     Class   4: IoU = 0.3193
INFO:     Class   5: IoU = 0.0000
INFO:     Class   6: IoU = 0.0433
INFO:     Class   7: IoU = 0.0046
INFO:     Class   8: IoU = 0.0000
INFO: Validation Dice score: 0.5618892908096313
INFO: Checkpoint 50 saved to results_ablation_C_sampler_full/checkpoints/checkpoint_epoch50.pth!
wandb: Waiting for W&B process to finish... (success).
wandb:
wandb: Run history:
wandb:           epoch ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:   learning rate ███████▂▂▂▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:            step ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇███
wandb:      train loss ▇▄▃▄▂▅▂▃▁▂▃▂▂▂▁▁▂▂▃▂▃▃▂▂▂▃█▂▃▃▆▂▂▃█▃▅▂▂▁
wandb: validation Dice ▄▂█▁▇▆▇▅█▇▇█▇██▇▇▇████████▇▇▇▇██▇█▇███▇█
wandb:
wandb: Run summary:
wandb:           epoch 50
wandb:   learning rate 0.0
wandb:            step 6300
wandb:      train loss 1.57576
wandb: validation Dice 0.56189
wandb:
wandb: You can sync this run to the cloud by running:
wandb: wandb sync /workspace/SSCloud/train/wandb/offline-run-20260413_171239-58lvtxek
wandb: Find logs at: ./wandb/offline-run-20260413_171239-58lvtxek/logs

DONE: [C_Sampler]

Stage duration: 00:27:48

# D Weights + oversampling
INFO: Starting training:
        Epochs:          50
        Batch size:      1
        Learning rate:   8e-05
        Training size:   126
        Validation size: 13
        Checkpoints:     True
        Device:          cuda
        Images scaling:  0.1
        Mixed Precision: False
        Num workers:     4
        Persistent wrk:  True

Detailed metrics saved to results_ablation_D_both_full
  - Confusion matrix: results_ablation_D_both_full/confusion_matrix_epoch50.png
  - Per-class IoU: results_ablation_D_both_full/per_class_iou_epoch50.txt
  - Average inference time per image: 0.0106 seconds
INFO: Final Epoch 50 - Detailed Evaluation:
INFO:   Dice Score: 0.3431
INFO:   Average Inference Time per image: 0.0106 seconds
INFO:   Mean IoU (excluding NaN): 0.2568
INFO:   Per-Class IoU (first 10 classes):
INFO:     Class   0: IoU = 0.1324
INFO:     Class   1: IoU = 0.6608
INFO:     Class   2: IoU = 0.5382
INFO:     Class   3: IoU = 0.5571
INFO:     Class   4: IoU = 0.3327
INFO:     Class   5: IoU = 0.0035
INFO:     Class   6: IoU = 0.0173
INFO:     Class   7: IoU = 0.0371
INFO:     Class   8: IoU = 0.0322
INFO: Validation Dice score: 0.3430544137954712
INFO: Checkpoint 50 saved to results_ablation_D_both_full/checkpoints/checkpoint_epoch50.pth!
wandb: Waiting for W&B process to finish... (success).
wandb:
wandb: Run history:
wandb:           epoch ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:   learning rate ██████▂▂▂▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:            step ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇███
wandb:      train loss ██▄▅▅▃▂▃▃▄▂▂▅▂▂▃▂▇▂▃▂▄▁▂▄▄▂▃▇▇▂▄▁▃▃▁▃▃▂▂
wandb: validation Dice ▁█▂▆▅▅▄▄▃▅▃▅▂▃▂▂▂▃▂▂▂▃▃▃▃▃▃▃▃▂▂▂▂▂▂▂▂▂▁▁
wandb:
wandb: Run summary:
wandb:           epoch 50
wandb:   learning rate 0.0
wandb:            step 6300
wandb:      train loss 1.19691
wandb: validation Dice 0.34305
wandb:
wandb: You can sync this run to the cloud by running:
wandb: wandb sync /workspace/SSCloud/train/wandb/offline-run-20260413_174037-63am339r
wandb: Find logs at: ./wandb/offline-run-20260413_174037-63am339r/logs

DONE: [D_Both]

Stage duration: 00:27:51

============================================================
ALL EXPERIMENTS COMPLETED
Total elapsed time: 01:59:27
============================================================


Посмотрел F:\Desktop\SSCloud\summary_run_ablation_full.md — краткий разбор:
- Лучший результат у A (Baseline): Dice 0.6450, mIoU 0.2921 (лучшие оба ключевых метрики).
- C (Oversampling only) второй: Dice 0.5619, mIoU 0.2670 — хуже baseline, но лучше, чем варианты с class weights.
- B (Class weights only) заметно просел: Dice 0.4132, mIoU 0.2532.
- D (Weights + Oversampling) худший по Dice: 0.3431, mIoU 0.2568 (чуть выше B по mIoU, но сильно хуже по Dice).
- Инференс почти одинаковый у всех (~0.0095–0.0109 s/img) — разница в качестве, не в скорости.
Что это значит по сути:
- В текущей конфигурации class weights скорее вредят глобальной сегментации.
- oversampling в одиночку помогает меньше, чем ожидалось, и не обгоняет baseline.
- Комбинация weights+sampler, похоже, переусиливает редкие классы и ломает общую стабильность (особенно видно по падению Dice).
Если хочешь, могу следующим шагом сделать точечный план тюнинга именно для редких классов (без потери Dice): более мягкие веса, clip/temperature для sampler, и отдельный ablation на focal loss/dice+ce баланс.