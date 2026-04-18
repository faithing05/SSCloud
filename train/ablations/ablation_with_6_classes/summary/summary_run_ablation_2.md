# A Baseline control
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

Detailed metrics saved to results_ablation2_A_baseline_ref_s010
  - Confusion matrix: results_ablation2_A_baseline_ref_s010/confusion_matrix_epoch20.png
  - Per-class IoU: results_ablation2_A_baseline_ref_s010/per_class_iou_epoch20.txt
  - Average inference time per image: 0.0098 seconds
INFO: Final Epoch 20 - Detailed Evaluation:
INFO:   Dice Score: 0.4630
INFO:   Average Inference Time per image: 0.0098 seconds
INFO:   Mean IoU (excluding NaN): 0.3943
INFO:   Per-Class IoU (first 10 classes):
INFO:     Class   0: IoU = 0.7466
INFO:     Class   1: IoU = 0.5578
INFO:     Class   2: IoU = 0.6228
INFO:     Class   3: IoU = 0.4358
INFO:     Class   4: IoU = 0.0005
INFO:     Class   5: IoU = 0.0021
INFO: Validation Dice score: 0.4629945755004883
INFO: Checkpoint 20 saved to results_ablation2_A_baseline_ref_s010/checkpoints/checkpoint_epoch20.pth!
wandb: Waiting for W&B process to finish... (success).
wandb:
wandb: Run history:
wandb:           epoch ▁▁▁▁▂▂▂▂▂▂▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▇▇▇▇▇▇████
wandb:   learning rate ██████████████▁▁▁▁▁▁
wandb:            step ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:      train loss ▄▃█▃▄▃▄▃▄▂▅▄▂▃▂▄▄▂▂▃▃▃▂▂▃▃▄▃▂▁▇▄▂▃▃▄▄▂▃▃
wandb: validation Dice ▁▆▇█▂▇▇██▆▅▆▇▄▅▅▄▅▅▆
wandb:
wandb: Run summary:
wandb:           epoch 20
wandb:   learning rate 1e-05
wandb:            step 2140
wandb:      train loss 2.8557
wandb: validation Dice 0.46299
wandb:
wandb: You can sync this run to the cloud by running:
wandb: wandb sync /workspace/SSCloud/train/wandb/offline-run-20260418_135311-mic00z18
wandb: Find logs at: ./wandb/offline-run-20260418_135311-mic00z18/logs

DONE: [A_Baseline_ref_s010]

Stage duration: 00:10:12

# B Sampler-only (slightly stronger than ultralight)
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

Detailed metrics saved to results_ablation2_B_sampler_balanced_s010
  - Confusion matrix: results_ablation2_B_sampler_balanced_s010/confusion_matrix_epoch20.png
  - Per-class IoU: results_ablation2_B_sampler_balanced_s010/per_class_iou_epoch20.txt
  - Average inference time per image: 0.0093 seconds
INFO: Final Epoch 20 - Detailed Evaluation:
INFO:   Dice Score: 0.4494
INFO:   Average Inference Time per image: 0.0093 seconds
INFO:   Mean IoU (excluding NaN): 0.3824
INFO:   Per-Class IoU (first 10 classes):
INFO:     Class   0: IoU = 0.7519
INFO:     Class   1: IoU = 0.5247
INFO:     Class   2: IoU = 0.6118
INFO:     Class   3: IoU = 0.4046
INFO:     Class   4: IoU = 0.0000
INFO:     Class   5: IoU = 0.0011
INFO: Validation Dice score: 0.44940897822380066
INFO: Checkpoint 20 saved to results_ablation2_B_sampler_balanced_s010/checkpoints/checkpoint_epoch20.pth!
wandb: Waiting for W&B process to finish... (success).
wandb:
wandb: Run history:
wandb:           epoch ▁▁▁▁▂▂▂▂▂▂▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▇▇▇▇▇▇████
wandb:   learning rate ██████████████▁▁▁▁▁▁
wandb:            step ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:      train loss ▄▄▃▅▅▄▃▅▃▂▅▂▃▂▃▅▅▃▂▄▁▂▄█▄▃▂▇▄▄▅▃▃▂▄▂▃▂▂▂
wandb: validation Dice ▄▄█▇▁▅▆██▅▅██▄▅▅▆▆█▆
wandb:
wandb: Run summary:
wandb:           epoch 20
wandb:   learning rate 1e-05
wandb:            step 2140
wandb:      train loss 0.93044
wandb: validation Dice 0.44941
wandb:
wandb: You can sync this run to the cloud by running:
wandb: wandb sync /workspace/SSCloud/train/wandb/offline-run-20260418_140326-5io5ld42
wandb: Find logs at: ./wandb/offline-run-20260418_140326-5io5ld42/logs

DONE: [B_Sampler_balanced_s010]

Stage duration: 00:09:52

# C Mild weights + very light sampler
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

Detailed metrics saved to results_ablation2_C_hybrid_mild_s010
  - Confusion matrix: results_ablation2_C_hybrid_mild_s010/confusion_matrix_epoch20.png
  - Per-class IoU: results_ablation2_C_hybrid_mild_s010/per_class_iou_epoch20.txt
  - Average inference time per image: 0.0087 seconds
INFO: Final Epoch 20 - Detailed Evaluation:
INFO:   Dice Score: 0.5080
INFO:   Average Inference Time per image: 0.0087 seconds
INFO:   Mean IoU (excluding NaN): 0.3873
INFO:   Per-Class IoU (first 10 classes):
INFO:     Class   0: IoU = 0.7512
INFO:     Class   1: IoU = 0.5314
INFO:     Class   2: IoU = 0.6217
INFO:     Class   3: IoU = 0.4194
INFO:     Class   4: IoU = 0.0000
INFO:     Class   5: IoU = 0.0000
INFO: Validation Dice score: 0.5080278515815735
INFO: Checkpoint 20 saved to results_ablation2_C_hybrid_mild_s010/checkpoints/checkpoint_epoch20.pth!
wandb: Waiting for W&B process to finish... (success).
wandb:
wandb: Run history:
wandb:           epoch ▁▁▁▁▂▂▂▂▂▂▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▇▇▇▇▇▇████
wandb:   learning rate █████████████▂▂▂▂▂▂▁
wandb:            step ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:      train loss ▇▃▆▃▅▄▄▃▄▂▄▂▃▂▂█▃▁▃▃▃▂▄▁▂▁▂▂▂▂▃▂▄▂▂▂▂▃▁▂
wandb: validation Dice ▁▆▄▄▆▆██▆▂▄▇▃▃▄▆▂▃▇▇
wandb:
wandb: Run summary:
wandb:           epoch 20
wandb:   learning rate 0.0
wandb:            step 2140
wandb:      train loss 0.98704
wandb: validation Dice 0.50803
wandb:
wandb: You can sync this run to the cloud by running:
wandb: wandb sync /workspace/SSCloud/train/wandb/offline-run-20260418_141312-7jglv1nb
wandb: Find logs at: ./wandb/offline-run-20260418_141312-7jglv1nb/logs

DONE: [C_Hybrid_mild_s010]

Stage duration: 00:10:00

# D Weights-only (bounded to reduce Dice drop risk)
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

Detailed metrics saved to results_ablation2_D_weights_mild_s010
  - Confusion matrix: results_ablation2_D_weights_mild_s010/confusion_matrix_epoch20.png
  - Per-class IoU: results_ablation2_D_weights_mild_s010/per_class_iou_epoch20.txt
  - Average inference time per image: 0.0091 seconds
INFO: Final Epoch 20 - Detailed Evaluation:
INFO:   Dice Score: 0.3842
INFO:   Average Inference Time per image: 0.0091 seconds
INFO:   Mean IoU (excluding NaN): 0.3881
INFO:   Per-Class IoU (first 10 classes):
INFO:     Class   0: IoU = 0.7444
INFO:     Class   1: IoU = 0.5358
INFO:     Class   2: IoU = 0.5990
INFO:     Class   3: IoU = 0.4439
INFO:     Class   4: IoU = 0.0053
INFO:     Class   5: IoU = 0.0000
INFO: Validation Dice score: 0.38422802090644836
INFO: Checkpoint 20 saved to results_ablation2_D_weights_mild_s010/checkpoints/checkpoint_epoch20.pth!
wandb: Waiting for W&B process to finish... (success).
wandb:
wandb: Run history:
wandb:           epoch ▁▁▁▁▂▂▂▂▂▂▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▇▇▇▇▇▇████
wandb:   learning rate ███████████▂▂▂▂▂▂▁▁▁
wandb:            step ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:      train loss ▃▃▂▃▂▃▂▂▃▄▂▃▅▄▃▂▃▃▃▄▃▅▃▃▂▃▅▁▄▁▃▂▂▁▄▃▃█▃▁
wandb: validation Dice ▁▄▁▁▇█▇▇▄▁▇▂▂▂▇▇▇▇▇▂
wandb:
wandb: Run summary:
wandb:           epoch 20
wandb:   learning rate 0.0
wandb:            step 2140
wandb:      train loss 0.49047
wandb: validation Dice 0.38423
wandb:
wandb: You can sync this run to the cloud by running:
wandb: wandb sync /workspace/SSCloud/train/wandb/offline-run-20260418_142311-u7t09bht
wandb: Find logs at: ./wandb/offline-run-20260418_142311-u7t09bht/logs

DONE: [D_Weights_mild_s010]

Stage duration: 00:09:41