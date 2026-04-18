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

Detailed metrics saved to results_quality_hybrid_sampler_plus
  - Confusion matrix: results_quality_hybrid_sampler_plus/confusion_matrix_epoch50.png
  - Per-class IoU: results_quality_hybrid_sampler_plus/per_class_iou_epoch50.txt
  - Average inference time per image: 0.0088 seconds
INFO: Final Epoch 50 - Detailed Evaluation:
INFO:   Dice Score: 0.3971
INFO:   Average Inference Time per image: 0.0088 seconds
INFO:   Mean IoU (excluding NaN): 0.3833
INFO:   Per-Class IoU (first 10 classes):
INFO:     Class   0: IoU = 0.7444
INFO:     Class   1: IoU = 0.5463
INFO:     Class   2: IoU = 0.5921
INFO:     Class   3: IoU = 0.4088
INFO:     Class   4: IoU = 0.0000
INFO:     Class   5: IoU = 0.0084
INFO: Validation Dice score: 0.3970527648925781
INFO: Checkpoint 50 saved to results_quality_hybrid_sampler_plus/checkpoints/checkpoint_epoch50.pth!
wandb: Waiting for W&B process to finish... (success).
wandb:
wandb: Run history:
wandb:           epoch ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:   learning rate ████████▂▂▂▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:            step ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇███
wandb:      train loss ▅▇▄▃▄█▂▄▅▆▃▃▅▄▄▂▃▁▃▂▂▃▄▃▃▆▄▆▂▂▆▂▄▃▄▂▂▅▄▇
wandb: validation Dice ▄▂▂▁█▆▅▃▃▅▆▇▅▅▅▅▅▅▅▅▄▄▅▅▆▅▅▅▅▅▅▅▅▅▅▆▅▅▅▅
wandb:
wandb: Run summary:
wandb:           epoch 50
wandb:   learning rate 0.0
wandb:            step 5350
wandb:      train loss 0.7411
wandb: validation Dice 0.39705
wandb:
wandb: You can sync this run to the cloud by running:
wandb: wandb sync /workspace/SSCloud/train/wandb/offline-run-20260418_165311-dogh1ns0
wandb: Find logs at: ./wandb/offline-run-20260418_165311-dogh1ns0/logs
Training finished. Best checkpoint is in: results_quality_hybrid_sampler_plus/checkpoints/
  End time:      2026-04-18 17:14:48
  Duration:      00:22:52