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

Detailed metrics saved to results_quality_best_ablation5_A
  - Confusion matrix: results_quality_best_ablation5_A/confusion_matrix_epoch50.png
  - Per-class IoU: results_quality_best_ablation5_A/per_class_iou_epoch50.txt
  - Average inference time per image: 0.0088 seconds
INFO: Final Epoch 50 - Detailed Evaluation:
INFO:   Dice Score: 0.4030
INFO:   Average Inference Time per image: 0.0088 seconds
INFO:   Mean IoU (excluding NaN): 0.3900
INFO:   Per-Class IoU (first 10 classes):
INFO:     Class   0: IoU = 0.7524
INFO:     Class   1: IoU = 0.5406
INFO:     Class   2: IoU = 0.6213
INFO:     Class   3: IoU = 0.4259
INFO:     Class   4: IoU = 0.0000
INFO:     Class   5: IoU = 0.0000
INFO: Validation Dice score: 0.4029592275619507
INFO: Checkpoint 50 saved to results_quality_best_ablation5_A/checkpoints/checkpoint_epoch50.pth!
wandb: Waiting for W&B process to finish... (success).
wandb:
wandb: Run history:
wandb:           epoch ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:   learning rate ████████████▂▂▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:            step ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇███
wandb:      train loss █▃▆▅▅▇▃▃▃▄▃▁▄▃▃▃▂▁▄▄▃▃▁▃▃▅▃▄▂▄▄▁▆▁▅▁▃▆▃▆
wandb: validation Dice ▁▁▆▃▇▇██▃▃▄▅▆▅▆▇▄▄▄▄▄▄▄▄▄▄▄▄▄▅▄▄▅▄▄▄▅▄▅▄
wandb:
wandb: Run summary:
wandb:           epoch 50
wandb:   learning rate 0.0
wandb:            step 5350
wandb:      train loss 1.87978
wandb: validation Dice 0.40296
wandb:
wandb: You can sync this run to the cloud by running:
wandb: wandb sync /workspace/SSCloud/train/wandb/offline-run-20260418_172712-ifloqwuv
wandb: Find logs at: ./wandb/offline-run-20260418_172712-ifloqwuv/logs
Training finished. Best checkpoint is in: results_quality_best_ablation5_A/checkpoints/
  End time:      2026-04-18 17:49:30
  Duration:      00:23:34