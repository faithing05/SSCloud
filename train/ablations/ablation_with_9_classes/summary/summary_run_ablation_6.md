# A Control: proven baseline.
INFO: Starting training:
        Epochs:          20
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

Detailed metrics saved to results_ablation6_A_baseline_ctrl
  - Confusion matrix: results_ablation6_A_baseline_ctrl/confusion_matrix_epoch20.png
  - Per-class IoU: results_ablation6_A_baseline_ctrl/per_class_iou_epoch20.txt
  - Average inference time per image: 0.0108 seconds
INFO: Final Epoch 20 - Detailed Evaluation:
INFO:   Dice Score: 0.5501
INFO:   Average Inference Time per image: 0.0108 seconds
INFO:   Mean IoU (excluding NaN): 0.3020
INFO:   Per-Class IoU (first 10 classes):
INFO:     Class   0: IoU = 0.2295
INFO:     Class   1: IoU = 0.9136
INFO:     Class   2: IoU = 0.5462
INFO:     Class   3: IoU = 0.5863
INFO:     Class   4: IoU = 0.3393
INFO:     Class   5: IoU = 0.0000
INFO:     Class   6: IoU = 0.1034
INFO:     Class   7: IoU = 0.0000
INFO:     Class   8: IoU = 0.0000
INFO: Validation Dice score: 0.5500895380973816
INFO: Checkpoint 20 saved to results_ablation6_A_baseline_ctrl/checkpoints/checkpoint_epoch20.pth!
wandb: Waiting for W&B process to finish... (success).
wandb:
wandb: Run history:
wandb:           epoch ▁▁▁▁▂▂▂▂▂▂▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▇▇▇▇▇▇████
wandb:   learning rate ███████▂▂▂▂▂▂▁▁▁▁▁▁▁
wandb:            step ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:      train loss █▂▄▅▃▄▁▂▄▅▂▂▇▁▂▄▃▃▁▁▁▂▂▂▂▂▄▂▃▂▁▁▂▄▄▄▁▂▂▂
wandb: validation Dice ▁██▅▆▇▆▆▇▆▆▇▇▇▇▆▇▆▆▆
wandb:
wandb: Run summary:
wandb:           epoch 20
wandb:   learning rate 0.0
wandb:            step 2520
wandb:      train loss 1.0213
wandb: validation Dice 0.55009
wandb:
wandb: You can sync this run to the cloud by running:
wandb: wandb sync /workspace/SSCloud/train/wandb/offline-run-20260413_204122-dvj02hi6
wandb: Find logs at: ./wandb/offline-run-20260413_204122-dvj02hi6/logs

DONE: [A_Baseline_ctrl]

Stage duration: 00:13:35

# B Very mild class-weights (test rare-class gain with minimal Dice risk).
INFO: Starting training:
        Epochs:          20
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

Detailed metrics saved to results_ablation6_B_weights_mild
  - Confusion matrix: results_ablation6_B_weights_mild/confusion_matrix_epoch20.png
  - Per-class IoU: results_ablation6_B_weights_mild/per_class_iou_epoch20.txt
  - Average inference time per image: 0.0106 seconds
INFO: Final Epoch 20 - Detailed Evaluation:
INFO:   Dice Score: 0.5671
INFO:   Average Inference Time per image: 0.0106 seconds
INFO:   Mean IoU (excluding NaN): 0.2844
INFO:   Per-Class IoU (first 10 classes):
INFO:     Class   0: IoU = 0.1542
INFO:     Class   1: IoU = 0.9241
INFO:     Class   2: IoU = 0.5391
INFO:     Class   3: IoU = 0.5347
INFO:     Class   4: IoU = 0.3665
INFO:     Class   5: IoU = 0.0000
INFO:     Class   6: IoU = 0.0412
INFO:     Class   7: IoU = 0.0000
INFO:     Class   8: IoU = 0.0000
INFO: Validation Dice score: 0.5670568943023682
INFO: Checkpoint 20 saved to results_ablation6_B_weights_mild/checkpoints/checkpoint_epoch20.pth!
wandb: Waiting for W&B process to finish... (success).
wandb:
wandb: Run history:
wandb:           epoch ▁▁▁▁▂▂▂▂▂▂▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▇▇▇▇▇▇████
wandb:   learning rate ███████████████▁▁▁▁▁
wandb:            step ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:      train loss ▆▃▆█▃▃▂▂▅▂▄▂▁▂▃▁▂▂▂▁▂▂▄▃▁▄▄▂▃▄▁▂▂▄▃▂▁▅▁▁
wandb: validation Dice ▃▁▄▅▃▅▆▆▆█▇▂▇▇▇██▇██
wandb:
wandb: Run summary:
wandb:           epoch 20
wandb:   learning rate 1e-05
wandb:            step 2520
wandb:      train loss 1.50039
wandb: validation Dice 0.56706
wandb:
wandb: You can sync this run to the cloud by running:
wandb: wandb sync /workspace/SSCloud/train/wandb/offline-run-20260413_205454-orc95m0s
wandb: Find logs at: ./wandb/offline-run-20260413_205454-orc95m0s/logs

DONE: [B_Weights_mild]

Stage duration: 00:13:47

# C Ultra-light sampler only (checks class-6 lift without strong distribution shift).
INFO: Starting training:
        Epochs:          20
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

Detailed metrics saved to results_ablation6_C_sampler_ultralight
  - Confusion matrix: results_ablation6_C_sampler_ultralight/confusion_matrix_epoch20.png
  - Per-class IoU: results_ablation6_C_sampler_ultralight/per_class_iou_epoch20.txt
  - Average inference time per image: 0.0118 seconds
INFO: Final Epoch 20 - Detailed Evaluation:
INFO:   Dice Score: 0.5620
INFO:   Average Inference Time per image: 0.0118 seconds
INFO:   Mean IoU (excluding NaN): 0.2913
INFO:   Per-Class IoU (first 10 classes):
INFO:     Class   0: IoU = 0.1811
INFO:     Class   1: IoU = 0.9036
INFO:     Class   2: IoU = 0.5311
INFO:     Class   3: IoU = 0.5655
INFO:     Class   4: IoU = 0.3725
INFO:     Class   5: IoU = 0.0000
INFO:     Class   6: IoU = 0.0400
INFO:     Class   7: IoU = 0.0281
INFO:     Class   8: IoU = 0.0000
INFO: Validation Dice score: 0.5619640350341797
INFO: Checkpoint 20 saved to results_ablation6_C_sampler_ultralight/checkpoints/checkpoint_epoch20.pth!
wandb: Waiting for W&B process to finish... (success).
wandb:
wandb: Run history:
wandb:           epoch ▁▁▁▁▂▂▂▂▂▂▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▇▇▇▇▇▇████
wandb:   learning rate ███████▂▂▂▂▂▂▂▂▂▁▁▁▁
wandb:            step ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:      train loss ▄█▃▃▅▄▄▃▃▅▄▇▂▂▂▅▃▂▄▅▂▃▁▄▃▁▃▁▁▂▇▂▆▆▂▂▁▄▅▂
wandb: validation Dice ▁▇▄▄▅▃▅▄▄▆█▇▅▅▅▇▅▅▅▅
wandb:
wandb: Run summary:
wandb:           epoch 20
wandb:   learning rate 0.0
wandb:            step 2520
wandb:      train loss 0.57096
wandb: validation Dice 0.56196
wandb:
wandb: You can sync this run to the cloud by running:
wandb: wandb sync /workspace/SSCloud/train/wandb/offline-run-20260413_210836-xfgbwi03
wandb: Find logs at: ./wandb/offline-run-20260413_210836-xfgbwi03/logs

DONE: [C_Sampler_ultralight]

Stage duration: 00:13:09

============================================================
ALL EXPERIMENTS COMPLETED
Total elapsed time: 00:40:31
============================================================