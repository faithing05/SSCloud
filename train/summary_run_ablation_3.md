# A Baseline
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

Detailed metrics saved to results_ablation_A_baseline_short
  - Confusion matrix: results_ablation_A_baseline_short/confusion_matrix_epoch10.png
  - Per-class IoU: results_ablation_A_baseline_short/per_class_iou_epoch10.txt
  - Average inference time per image: 0.0085 seconds
INFO: Final Epoch 10 - Detailed Evaluation:
INFO:   Dice Score: 0.5906
INFO:   Average Inference Time per image: 0.0085 seconds
INFO:   Mean IoU (excluding NaN): 0.2264
INFO:   Per-Class IoU (first 10 classes):
INFO:     Class   0: IoU = 0.1040
INFO:     Class   1: IoU = 0.9187
INFO:     Class   2: IoU = 0.1120
INFO:     Class   3: IoU = 0.5349
INFO:     Class   4: IoU = 0.3679
INFO:     Class   5: IoU = 0.0000
INFO:     Class   6: IoU = 0.0000
INFO:     Class   7: IoU = 0.0000
INFO:     Class   8: IoU = 0.0000
INFO: Validation Dice score: 0.5905701518058777
INFO: Checkpoint 10 saved to results_ablation_A_baseline_short/checkpoints/checkpoint_epoch10.pth!
wandb: Waiting for W&B process to finish... (success).
wandb:
wandb: Run history:
wandb:           epoch ▁▁▁▁▂▂▂▂▃▃▃▃▃▃▃▃▄▄▄▄▅▅▅▅▆▆▆▆▆▆▆▆▇▇▇▇████
wandb:   learning rate ▁▁▁▁▁▁▁▁▁▁
wandb:            step ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:      train loss ▅▄▂▆▃▃▅▆▄▆▃▆▂▃▁▂▃▂▂▂▃▃▄▁▂▅▄▆▂▅▇▄▄▃▅▃▁█▃▂
wandb: validation Dice ▂▁▆▆▇▇▇▇▆█
wandb:
wandb: Run summary:
wandb:           epoch 10
wandb:   learning rate 0.0001
wandb:            step 1260
wandb:      train loss 2.26407
wandb: validation Dice 0.59057
wandb:
wandb: You can sync this run to the cloud by running:
wandb: wandb sync /workspace/SSCloud/train/wandb/offline-run-20260413_182034-mpwb88wd
wandb: Find logs at: ./wandb/offline-run-20260413_182034-mpwb88wd/logs

DONE: [A_Baseline]

Stage duration: 00:07:28

# B Class weights only
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

Detailed metrics saved to results_ablation_B_weights_soft_short
  - Confusion matrix: results_ablation_B_weights_soft_short/confusion_matrix_epoch10.png
  - Per-class IoU: results_ablation_B_weights_soft_short/per_class_iou_epoch10.txt
  - Average inference time per image: 0.0095 seconds
INFO: Final Epoch 10 - Detailed Evaluation:
INFO:   Dice Score: 0.5607
INFO:   Average Inference Time per image: 0.0095 seconds
INFO:   Mean IoU (excluding NaN): 0.2878
INFO:   Per-Class IoU (first 10 classes):
INFO:     Class   0: IoU = 0.1719
INFO:     Class   1: IoU = 0.9130
INFO:     Class   2: IoU = 0.5495
INFO:     Class   3: IoU = 0.5325
INFO:     Class   4: IoU = 0.3589
INFO:     Class   5: IoU = 0.0000
INFO:     Class   6: IoU = 0.0644
INFO:     Class   7: IoU = 0.0000
INFO:     Class   8: IoU = 0.0000
INFO: Validation Dice score: 0.5606899261474609
INFO: Checkpoint 10 saved to results_ablation_B_weights_soft_short/checkpoints/checkpoint_epoch10.pth!
wandb: Waiting for W&B process to finish... (success).
wandb:
wandb: Run history:
wandb:           epoch ▁▁▁▁▂▂▂▂▃▃▃▃▃▃▃▃▄▄▄▄▅▅▅▅▆▆▆▆▆▆▆▆▇▇▇▇████
wandb:   learning rate ██████▁▁▁▁
wandb:            step ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:      train loss ▄▄▂▆▄▄█▃▃▃▅▃▂▂▂▃▂▂▃▄▂▆▅▄▃▃▁▅▃▄▂▄▂▂▁▁▆▄▂▃
wandb: validation Dice █▁▁▆▅▅▃▅▅▆
wandb:
wandb: Run summary:
wandb:           epoch 10
wandb:   learning rate 1e-05
wandb:            step 1260
wandb:      train loss 0.98705
wandb: validation Dice 0.56069
wandb:
wandb: You can sync this run to the cloud by running:
wandb: wandb sync /workspace/SSCloud/train/wandb/offline-run-20260413_182801-s19n70a7
wandb: Find logs at: ./wandb/offline-run-20260413_182801-s19n70a7/logs

DONE: [B_WeightsSoft]

Stage duration: 00:07:30

# C Oversampling only
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

Detailed metrics saved to results_ablation_C_sampler_soft_short
  - Confusion matrix: results_ablation_C_sampler_soft_short/confusion_matrix_epoch10.png
  - Per-class IoU: results_ablation_C_sampler_soft_short/per_class_iou_epoch10.txt
  - Average inference time per image: 0.0088 seconds
INFO: Final Epoch 10 - Detailed Evaluation:
INFO:   Dice Score: 0.4930
INFO:   Average Inference Time per image: 0.0088 seconds
INFO:   Mean IoU (excluding NaN): 0.2558
INFO:   Per-Class IoU (first 10 classes):
INFO:     Class   0: IoU = 0.0986
INFO:     Class   1: IoU = 0.9124
INFO:     Class   2: IoU = 0.4857
INFO:     Class   3: IoU = 0.5420
INFO:     Class   4: IoU = 0.2435
INFO:     Class   5: IoU = 0.0000
INFO:     Class   6: IoU = 0.0007
INFO:     Class   7: IoU = 0.0193
INFO:     Class   8: IoU = 0.0000
INFO: Validation Dice score: 0.49304062128067017
INFO: Checkpoint 10 saved to results_ablation_C_sampler_soft_short/checkpoints/checkpoint_epoch10.pth!
wandb: Waiting for W&B process to finish... (success).
wandb:
wandb: Run history:
wandb:           epoch ▁▁▁▁▂▂▂▂▃▃▃▃▃▃▃▃▄▄▄▄▅▅▅▅▆▆▆▆▆▆▆▆▇▇▇▇████
wandb:   learning rate ▁▁▁▁▁▁▁▁▁▁
wandb:            step ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:      train loss ▄▄▂▄▂▃▅▅▂▄▂▃▄▂▄▃▃▆▄▃█▃▂▄▃▄▄▄▂▄▂▁▄▄▂▃▁▅▃▂
wandb: validation Dice ▅▃▅▇█▇▁▅▇▃
wandb:
wandb: Run summary:
wandb:           epoch 10
wandb:   learning rate 0.0001
wandb:            step 1260
wandb:      train loss 1.38753
wandb: validation Dice 0.49304
wandb:
wandb: You can sync this run to the cloud by running:
wandb: wandb sync /workspace/SSCloud/train/wandb/offline-run-20260413_183533-y74n1gr0
wandb: Find logs at: ./wandb/offline-run-20260413_183533-y74n1gr0/logs

DONE: [C_SamplerSoft]

Stage duration: 00:07:31

# D Weights + oversampling
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

Detailed metrics saved to results_ablation_D_both_soft_short
  - Confusion matrix: results_ablation_D_both_soft_short/confusion_matrix_epoch10.png
  - Per-class IoU: results_ablation_D_both_soft_short/per_class_iou_epoch10.txt
  - Average inference time per image: 0.0133 seconds
INFO: Final Epoch 10 - Detailed Evaluation:
INFO:   Dice Score: 0.5025
INFO:   Average Inference Time per image: 0.0133 seconds
INFO:   Mean IoU (excluding NaN): 0.2563
INFO:   Per-Class IoU (first 10 classes):
INFO:     Class   0: IoU = 0.1094
INFO:     Class   1: IoU = 0.9031
INFO:     Class   2: IoU = 0.5546
INFO:     Class   3: IoU = 0.4309
INFO:     Class   4: IoU = 0.3002
INFO:     Class   5: IoU = 0.0000
INFO:     Class   6: IoU = 0.0084
INFO:     Class   7: IoU = 0.0004
INFO:     Class   8: IoU = 0.0000
INFO: Validation Dice score: 0.5025021433830261
INFO: Checkpoint 10 saved to results_ablation_D_both_soft_short/checkpoints/checkpoint_epoch10.pth!
wandb: Waiting for W&B process to finish... (success).
wandb:
wandb: Run history:
wandb:           epoch ▁▁▁▁▂▂▂▂▃▃▃▃▃▃▃▃▄▄▄▄▅▅▅▅▆▆▆▆▆▆▆▆▇▇▇▇████
wandb:   learning rate ▁▁▁▁▁▁▁▁▁▁
wandb:            step ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:      train loss ▅▃▅▃▄▅▆▆▂▃▃█▄▆▄▂▅▂▂▄▂▂▃▆▃▆▅▁▆▄▄▄▃▄▃▂▄▁▅▆
wandb: validation Dice ▁▄▅▃█▅██▅▄
wandb:
wandb: Run summary:
wandb:           epoch 10
wandb:   learning rate 0.0001
wandb:            step 1260
wandb:      train loss 1.40379
wandb: validation Dice 0.5025
wandb:
wandb: You can sync this run to the cloud by running:
wandb: wandb sync /workspace/SSCloud/train/wandb/offline-run-20260413_184300-smrpuhwo
wandb: Find logs at: ./wandb/offline-run-20260413_184300-smrpuhwo/logs

DONE: [D_BothSoft]

Stage duration: 00:07:23

============================================================
ALL EXPERIMENTS COMPLETED
Total elapsed time: 00:29:52
============================================================

Отличный прогон, данные уже достаточно показательные.
- По Dice лидер снова A_Baseline: 0.5906 (summary_run_ablation_3.md:20).
- По mIoU лидер B_WeightsSoft: 0.2878 (summary_run_ablation_3.md:80) — это заметно выше baseline (0.2264).
- C_SamplerSoft и D_BothSoft проигрывают baseline по Dice (0.4930 и 0.5025), при этом по mIoU примерно на уровне друг друга (~0.256).
- По скорости инференса A/B/C близки (0.0085–0.0095 s/img), D медленнее (0.0133 s/img).
Быстрый вывод по стратегиям:
- Если главный KPI — глобальная сегментация (Dice), на 10 эпохах лучше Baseline.
- Если важнее баланс по классам (mIoU/редкие классы), сейчас лучше выглядит WeightsSoft.
- Oversampling (даже мягкий) пока не дает выигрыш по итоговой картине.
Что бы я делал дальше на full:
1. Прогнать full только A_Baseline и B_WeightsSoft (2 кандидата достаточно).
2. Для B добавить чуть мягче веса: --class-weight-power 0.4 (сейчас 0.5), оставить min/max как есть.
3. D/C пока не тратить GPU-время, если цель — максимум Dice при адекватном mIoU.
Если хочешь, могу сразу подготовить отдельный run_ablation_full_top2.sh (A vs B_soft + B_soft_p0.4), чтобы на удаленной машине запустить одним файлом.