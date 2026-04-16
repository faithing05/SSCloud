# A Reference baseline
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

Detailed metrics saved to results_ablation7_A_baseline_ref
  - Confusion matrix: results_ablation7_A_baseline_ref/confusion_matrix_epoch20.png
  - Per-class IoU: results_ablation7_A_baseline_ref/per_class_iou_epoch20.txt
  - Average inference time per image: 0.0080 seconds
INFO: Final Epoch 20 - Detailed Evaluation:
INFO:   Dice Score: 0.5683
INFO:   Average Inference Time per image: 0.0080 seconds
INFO:   Mean IoU (excluding NaN): 0.2796
INFO:   Per-Class IoU (first 10 classes):
INFO:     Class   0: IoU = 0.1079
INFO:     Class   1: IoU = 0.9214
INFO:     Class   2: IoU = 0.5386
INFO:     Class   3: IoU = 0.5346
INFO:     Class   4: IoU = 0.3487
INFO:     Class   5: IoU = 0.0000
INFO:     Class   6: IoU = 0.0651
INFO:     Class   7: IoU = 0.0000
INFO:     Class   8: IoU = 0.0000
INFO: Validation Dice score: 0.5682840943336487
INFO: Checkpoint 20 saved to results_ablation7_A_baseline_ref/checkpoints/checkpoint_epoch20.pth!
wandb: Waiting for W&B process to finish... (success).
wandb:
wandb: Run history:
wandb:           epoch ▁▁▁▁▂▂▂▂▂▂▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▇▇▇▇▇▇████
wandb:   learning rate ██████████████▁▁▁▁▁▁
wandb:            step ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:      train loss ▄█▂▄▃▅▂▂▄▆▂▄▃▄▇▅▃█▄▆▂▃▂▂▂▄▂▁▁▅▄▇▁▂▂▁▃▂▄▃
wandb: validation Dice ▃▂▇▆▁▃▆▇█▇▇▅▇▆▇▇█▇██
wandb:
wandb: Run summary:
wandb:           epoch 20
wandb:   learning rate 1e-05
wandb:            step 2520
wandb:      train loss 1.24276
wandb: validation Dice 0.56828
wandb:
wandb: You can sync this run to the cloud by running:
wandb: wandb sync /workspace/SSCloud/train/wandb/offline-run-20260416_135221-reoh0p2s
wandb: Find logs at: ./wandb/offline-run-20260416_135221-reoh0p2s/logs

DONE: [A_Baseline_ref]

Stage duration: 00:12:27

# B Rare-class sampler (stronger than ultralight, still bounded)
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

Detailed metrics saved to results_ablation7_B_sampler_balanced
  - Confusion matrix: results_ablation7_B_sampler_balanced/confusion_matrix_epoch20.png
  - Per-class IoU: results_ablation7_B_sampler_balanced/per_class_iou_epoch20.txt
  - Average inference time per image: 0.0071 seconds
INFO: Final Epoch 20 - Detailed Evaluation:
INFO:   Dice Score: 0.5495
INFO:   Average Inference Time per image: 0.0071 seconds
INFO:   Mean IoU (excluding NaN): 0.2884
INFO:   Per-Class IoU (first 10 classes):
INFO:     Class   0: IoU = 0.2014
INFO:     Class   1: IoU = 0.9192
INFO:     Class   2: IoU = 0.5342
INFO:     Class   3: IoU = 0.5059
INFO:     Class   4: IoU = 0.3616
INFO:     Class   5: IoU = 0.0000
INFO:     Class   6: IoU = 0.0730
INFO:     Class   7: IoU = 0.0000
INFO:     Class   8: IoU = 0.0000
INFO: Validation Dice score: 0.5494756698608398
INFO: Checkpoint 20 saved to results_ablation7_B_sampler_balanced/checkpoints/checkpoint_epoch20.pth!
wandb: Waiting for W&B process to finish... (success).
wandb:
wandb: Run history:
wandb:           epoch ▁▁▁▁▂▂▂▂▂▂▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▇▇▇▇▇▇████
wandb:   learning rate ████████████████▁▁▁▁
wandb:            step ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:      train loss ▄▂▁▂▃▃█▃▅▂▄▄▃▂▃▃▂▄▄▃▃▂▂▅▁▃▃▁▃▁▆▄▂▄▂▅▃▅▂▁
wandb: validation Dice ▁▆▅▆▄▆▇▆▇▆█▆▆▆▇▅▆▆▆▆
wandb:
wandb: Run summary:
wandb:           epoch 20
wandb:   learning rate 1e-05
wandb:            step 2520
wandb:      train loss 1.19653
wandb: validation Dice 0.54948
wandb:
wandb: You can sync this run to the cloud by running:
wandb: wandb sync /workspace/SSCloud/train/wandb/offline-run-20260416_140451-2bmgdatf
wandb: Find logs at: ./wandb/offline-run-20260416_140451-2bmgdatf/logs

DONE: [B_Sampler_balanced]

Stage duration: 00:12:29

# C Hybrid mild weights + light sampler
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

Detailed metrics saved to results_ablation7_C_hybrid_mild
  - Confusion matrix: results_ablation7_C_hybrid_mild/confusion_matrix_epoch20.png
  - Per-class IoU: results_ablation7_C_hybrid_mild/per_class_iou_epoch20.txt
  - Average inference time per image: 0.0075 seconds
INFO: Final Epoch 20 - Detailed Evaluation:
INFO:   Dice Score: 0.5098
INFO:   Average Inference Time per image: 0.0075 seconds
INFO:   Mean IoU (excluding NaN): 0.2765
INFO:   Per-Class IoU (first 10 classes):
INFO:     Class   0: IoU = 0.1453
INFO:     Class   1: IoU = 0.9154
INFO:     Class   2: IoU = 0.5207
INFO:     Class   3: IoU = 0.5303
INFO:     Class   4: IoU = 0.3587
INFO:     Class   5: IoU = 0.0000
INFO:     Class   6: IoU = 0.0180
INFO:     Class   7: IoU = 0.0000
INFO:     Class   8: IoU = 0.0000
INFO: Validation Dice score: 0.5098093152046204
INFO: Checkpoint 20 saved to results_ablation7_C_hybrid_mild/checkpoints/checkpoint_epoch20.pth!
wandb: Waiting for W&B process to finish... (success).
wandb:
wandb: Run history:
wandb:           epoch ▁▁▁▁▂▂▂▂▂▂▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▇▇▇▇▇▇████
wandb:   learning rate ██████████▂▂▂▂▂▂▁▁▁▁
wandb:            step ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:      train loss ▃▃▂▄▅▃▇▂▃▂▂▃▂▂▃▁▁▂▄▂▄▂▁▅▁▅▃▂▂▄▄▁▁▁█▂▁▃▂▂
wandb: validation Dice ▁██▇█▆▇▆▇▆▁█▆▇▆▆▆▆▆▇
wandb:
wandb: Run summary:
wandb:           epoch 20
wandb:   learning rate 0.0
wandb:            step 2520
wandb:      train loss 0.8282
wandb: validation Dice 0.50981
wandb:
wandb: You can sync this run to the cloud by running:
wandb: wandb sync /workspace/SSCloud/train/wandb/offline-run-20260416_141716-6it9r7dt
wandb: Find logs at: ./wandb/offline-run-20260416_141716-6it9r7dt/logs

DONE: [C_Hybrid_mild]

Stage duration: 00:12:31

============================================================
ALL EXPERIMENTS COMPLETED
Total elapsed time: 00:37:27
============================================================