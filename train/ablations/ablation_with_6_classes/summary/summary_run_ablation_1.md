# A Reference baseline (short-run control)
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

Detailed metrics saved to results_ablation1_A_baseline_ref_s010
  - Confusion matrix: results_ablation1_A_baseline_ref_s010/confusion_matrix_epoch20.png
  - Per-class IoU: results_ablation1_A_baseline_ref_s010/per_class_iou_epoch20.txt
  - Average inference time per image: 0.0086 seconds
INFO: Final Epoch 20 - Detailed Evaluation:
INFO:   Dice Score: 0.4320
INFO:   Average Inference Time per image: 0.0086 seconds
INFO:   Mean IoU (excluding NaN): 0.3815
INFO:   Per-Class IoU (first 10 classes):
INFO:     Class   0: IoU = 0.7478
INFO:     Class   1: IoU = 0.5499
INFO:     Class   2: IoU = 0.5841
INFO:     Class   3: IoU = 0.3943
INFO:     Class   4: IoU = 0.0130
INFO:     Class   5: IoU = 0.0000
INFO: Validation Dice score: 0.4320445656776428
INFO: Checkpoint 20 saved to results_ablation1_A_baseline_ref_s010/checkpoints/checkpoint_epoch20.pth!
wandb: Waiting for W&B process to finish... (success).
wandb:
wandb: Run history:
wandb:           epoch ▁▁▁▁▂▂▂▂▂▂▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▇▇▇▇▇▇████
wandb:   learning rate ██████████▂▂▂▂▂▂▁▁▁▁
wandb:            step ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:      train loss ▂▂▂█▃▂▂▃▂▁▁▂▂▂▂▂▁▂▅▂▂▁▅▂▂▂▁▂▂▁▁▂▁▅▂▂▁▁▂▁
wandb: validation Dice ▇█▁▄██▃▃▄▂▃▄▄▃▅▂▄▃▄▄
wandb:
wandb: Run summary:
wandb:           epoch 20
wandb:   learning rate 0.0
wandb:            step 2140
wandb:      train loss 2.04791
wandb: validation Dice 0.43204
wandb:
wandb: You can sync this run to the cloud by running:
wandb: wandb sync /workspace/SSCloud/train/wandb/offline-run-20260418_125809-99q59v1x
wandb: Find logs at: ./wandb/offline-run-20260418_125809-99q59v1x/logs

DONE: [A_Baseline_ref_s010]

Stage duration: 00:09:40

# B Mild class weights only
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

Detailed metrics saved to results_ablation1_B_weights_mild_s010
  - Confusion matrix: results_ablation1_B_weights_mild_s010/confusion_matrix_epoch20.png
  - Per-class IoU: results_ablation1_B_weights_mild_s010/per_class_iou_epoch20.txt
  - Average inference time per image: 0.0083 seconds
INFO: Final Epoch 20 - Detailed Evaluation:
INFO:   Dice Score: 0.4117
INFO:   Average Inference Time per image: 0.0083 seconds
INFO:   Mean IoU (excluding NaN): 0.3965
INFO:   Per-Class IoU (first 10 classes):
INFO:     Class   0: IoU = 0.7517
INFO:     Class   1: IoU = 0.5546
INFO:     Class   2: IoU = 0.6327
INFO:     Class   3: IoU = 0.4275
INFO:     Class   4: IoU = 0.0128
INFO:     Class   5: IoU = 0.0000
INFO: Validation Dice score: 0.4116840362548828
INFO: Checkpoint 20 saved to results_ablation1_B_weights_mild_s010/checkpoints/checkpoint_epoch20.pth!
wandb: Waiting for W&B process to finish... (success).
wandb:
wandb: Run history:
wandb:           epoch ▁▁▁▁▂▂▂▂▂▂▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▇▇▇▇▇▇████
wandb:   learning rate ███████▂▂▂▂▂▂▁▁▁▁▁▁▁
wandb:            step ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:      train loss ▄▃▃▃▄▃▃▃▂▁▃▂▂▄▃▂▂▂▂▂▂█▄▂▆▁▃▁▂▁▂▇▃▃▃▂▂▁▄▁
wandb: validation Dice ▃█▁▁█▅▁▆▆▆▁▁▆▆▇▂▂▂▂▃
wandb:
wandb: Run summary:
wandb:           epoch 20
wandb:   learning rate 0.0
wandb:            step 2140
wandb:      train loss 1.07048
wandb: validation Dice 0.41168
wandb:
wandb: You can sync this run to the cloud by running:
wandb: wandb sync /workspace/SSCloud/train/wandb/offline-run-20260418_130745-kqgwmuls
wandb: Find logs at: ./wandb/offline-run-20260418_130745-kqgwmuls/logs

DONE: [B_Weights_mild_s010]

Stage duration: 00:09:41

# C Ultra-light oversampling only
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

Detailed metrics saved to results_ablation1_C_sampler_ultralight_s010
  - Confusion matrix: results_ablation1_C_sampler_ultralight_s010/confusion_matrix_epoch20.png
  - Per-class IoU: results_ablation1_C_sampler_ultralight_s010/per_class_iou_epoch20.txt
  - Average inference time per image: 0.0085 seconds
INFO: Final Epoch 20 - Detailed Evaluation:
INFO:   Dice Score: 0.5456
INFO:   Average Inference Time per image: 0.0085 seconds
INFO:   Mean IoU (excluding NaN): 0.3877
INFO:   Per-Class IoU (first 10 classes):
INFO:     Class   0: IoU = 0.7514
INFO:     Class   1: IoU = 0.5360
INFO:     Class   2: IoU = 0.6173
INFO:     Class   3: IoU = 0.4217
INFO:     Class   4: IoU = 0.0000
INFO:     Class   5: IoU = 0.0000
INFO: Validation Dice score: 0.5455567836761475
INFO: Checkpoint 20 saved to results_ablation1_C_sampler_ultralight_s010/checkpoints/checkpoint_epoch20.pth!
wandb: Waiting for W&B process to finish... (success).
wandb:
wandb: Run history:
wandb:           epoch ▁▁▁▁▂▂▂▂▂▂▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▇▇▇▇▇▇████
wandb:   learning rate ███████████████▁▁▁▁▁
wandb:            step ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:      train loss ▂▅▂▄▄▃█▂▄▂▄▃▃▃▁█▄▂▂▃▂▂▂▃▂▁▃▄▆▂▂▃▃▂▃▅▃▆▄▂
wandb: validation Dice ▅▆▇▆▅▇▇▁▁██▂▂▇▂▃████
wandb:
wandb: Run summary:
wandb:           epoch 20
wandb:   learning rate 1e-05
wandb:            step 2140
wandb:      train loss 1.12878
wandb: validation Dice 0.54556
wandb:
wandb: You can sync this run to the cloud by running:
wandb: wandb sync /workspace/SSCloud/train/wandb/offline-run-20260418_131725-f4a28pu2
wandb: Find logs at: ./wandb/offline-run-20260418_131725-f4a28pu2/logs

DONE: [C_Sampler_ultralight_s010]

Stage duration: 00:09:39

# D Scale-up + mild weights
INFO: Starting training:
        Epochs:          20
        Batch size:      1
        Learning rate:   5e-05
        Training size:   107
        Validation size: 11
        Checkpoints:     True
        Device:          cuda
        Images scaling:  0.2
        Mixed Precision: False
        Num workers:     4
        Persistent wrk:  True

Epoch 1/20:   7%|████▌                                                                | 7/107 [00:06<01:29,  1.12img/s, loss (batch)=1.83]
ERROR: Detected OutOfMemoryError during training. Applying memory fallback: empty CUDA cache, enable checkpointing if available, and retry with AMP.
WARNING: Checkpointing fallback is not available for model type HybridSSCloudUNet; retrying without checkpointing.
INFO: Retrying with AMP enabled for lower memory usage
wandb: Waiting for W&B process to finish... (success).
wandb:
wandb: Run history:
wandb:      epoch ▁▁▁▁▁▁▁
wandb:       step ▁▂▃▅▆▇█
wandb: train loss █▅▄▁▂▁▁
wandb:
wandb: Run summary:
wandb:      epoch 1
wandb:       step 7
wandb: train loss 1.82993
wandb:
wandb: You can sync this run to the cloud by running:
wandb: wandb sync /workspace/SSCloud/train/wandb/offline-run-20260418_132652-5r2gtnhk
wandb: Find logs at: ./wandb/offline-run-20260418_132652-5r2gtnhk/logs
INFO: Creating dataset with 118 examples
INFO: Scanning mask files to determine unique values
100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 118/118 [00:18<00:00,  6.29it/s]
INFO: Unique mask values: [0, 2, 5, 6, 65, 66]
INFO: Class distribution report saved to results_ablation1_D_weights_mild_s020/class_distribution.txt
INFO: Using weighted CrossEntropyLoss with weights: [0.6539999842643738, 0.8331999778747559, 0.9042999744415283, 0.9021000266075134, 1.0807000398635864, 1.6258000135421753]
wandb: WARNING `resume` will be ignored since W&B syncing is set to `offline`. Starting a new run with run id is47vthj.
wandb: Tracking run with wandb version 0.13.5
wandb: W&B syncing is set to `offline` in this directory.
wandb: Run `wandb online` or set WANDB_MODE=online to enable cloud syncing.
INFO: Starting training:
        Epochs:          20
        Batch size:      1
        Learning rate:   5e-05
        Training size:   107
        Validation size: 11
        Checkpoints:     True
        Device:          cuda
        Images scaling:  0.2
        Mixed Precision: True
        Num workers:     4
        Persistent wrk:  True

Epoch 1/20:   0%|                                                                                                | 0/107 [00:01<?, ?img/s]
ERROR: OOM persisted after fallback (checkpointing=False, amp=True). Try smaller --scale, disable transformer (--no-transformer), or use a smaller batch size.