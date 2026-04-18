# A Reference best from ablation 3
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

Detailed metrics saved to results_ablation4_A_hybrid_sampler_ref
  - Confusion matrix: results_ablation4_A_hybrid_sampler_ref/confusion_matrix_epoch20.png
  - Per-class IoU: results_ablation4_A_hybrid_sampler_ref/per_class_iou_epoch20.txt
  - Average inference time per image: 0.0104 seconds
INFO: Final Epoch 20 - Detailed Evaluation:
INFO:   Dice Score: 0.5078
INFO:   Average Inference Time per image: 0.0104 seconds
INFO:   Mean IoU (excluding NaN): 0.3853
INFO:   Per-Class IoU (first 10 classes):
INFO:     Class   0: IoU = 0.7491
INFO:     Class   1: IoU = 0.5434
INFO:     Class   2: IoU = 0.6237
INFO:     Class   3: IoU = 0.3958
INFO:     Class   4: IoU = 0.0000
INFO:     Class   5: IoU = 0.0000
INFO: Validation Dice score: 0.5078089237213135
INFO: Checkpoint 20 saved to results_ablation4_A_hybrid_sampler_ref/checkpoints/checkpoint_epoch20.pth!
wandb: Waiting for W&B process to finish... (success).
wandb:
wandb: Run history:
wandb:           epoch ▁▁▁▁▂▂▂▂▂▂▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▇▇▇▇▇▇████
wandb:   learning rate ██████████▁▁▁▁▁▁▁▁▁▁
wandb:            step ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:      train loss ▅▄▃▂▂▃▂▃▂▄▄▂▄▄▂▄▃█▃▆▃▁▂▂▂▃▁▃▃▃▃▃▁▆▄▂▂▂▂▂
wandb: validation Dice ▅▇▁▄██▇▆▂▄▇█▆███▅▆▅▆
wandb:
wandb: Run summary:
wandb:           epoch 20
wandb:   learning rate 1e-05
wandb:            step 2140
wandb:      train loss 1.518
wandb: validation Dice 0.50781
wandb:
wandb: You can sync this run to the cloud by running:
wandb: wandb sync /workspace/SSCloud/train/wandb/offline-run-20260418_152102-70livcth
wandb: Find logs at: ./wandb/offline-run-20260418_152102-70livcth/logs

DONE: [A_Hybrid_sampler_ref]

Stage duration: 00:09:49

# B Slightly weaker sampler strength (stability check)
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

Detailed metrics saved to results_ablation4_B_hybrid_sampler_s012
  - Confusion matrix: results_ablation4_B_hybrid_sampler_s012/confusion_matrix_epoch20.png
  - Per-class IoU: results_ablation4_B_hybrid_sampler_s012/per_class_iou_epoch20.txt
  - Average inference time per image: 0.0093 seconds
INFO: Final Epoch 20 - Detailed Evaluation:
INFO:   Dice Score: 0.4224
INFO:   Average Inference Time per image: 0.0093 seconds
INFO:   Mean IoU (excluding NaN): 0.3910
INFO:   Per-Class IoU (first 10 classes):
INFO:     Class   0: IoU = 0.7445
INFO:     Class   1: IoU = 0.5591
INFO:     Class   2: IoU = 0.6210
INFO:     Class   3: IoU = 0.4212
INFO:     Class   4: IoU = 0.0000
INFO:     Class   5: IoU = 0.0000
INFO: Validation Dice score: 0.42236945033073425
INFO: Checkpoint 20 saved to results_ablation4_B_hybrid_sampler_s012/checkpoints/checkpoint_epoch20.pth!
wandb: Waiting for W&B process to finish... (success).
wandb:
wandb: Run history:
wandb:           epoch ▁▁▁▁▂▂▂▂▂▂▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▇▇▇▇▇▇████
wandb:   learning rate ███████▂▂▂▂▂▂▁▁▁▁▁▁▁
wandb:            step ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:      train loss ▂▅▃▃▃▄▃▃▃▂▃▃█▂▃▁▂▂▃▅▇▁▃▁▄▄▄▄▂▂▁▃▄▃▃▃▁▁▁▁
wandb: validation Dice ▅█▂▂▄▅▁▅▄▄▃▃▃▄▄▄▄▄▄▄
wandb:
wandb: Run summary:
wandb:           epoch 20
wandb:   learning rate 0.0
wandb:            step 2140
wandb:      train loss 1.00151
wandb: validation Dice 0.42237
wandb:
wandb: You can sync this run to the cloud by running:
wandb: wandb sync /workspace/SSCloud/train/wandb/offline-run-20260418_153048-udo7rk7l
wandb: Find logs at: ./wandb/offline-run-20260418_153048-udo7rk7l/logs

DONE: [B_Hybrid_sampler_s012]

Stage duration: 00:09:52

# C Slightly stronger sampler strength (rare-class push)
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

Detailed metrics saved to results_ablation4_C_hybrid_sampler_s018
  - Confusion matrix: results_ablation4_C_hybrid_sampler_s018/confusion_matrix_epoch20.png
  - Per-class IoU: results_ablation4_C_hybrid_sampler_s018/per_class_iou_epoch20.txt
  - Average inference time per image: 0.0098 seconds
INFO: Final Epoch 20 - Detailed Evaluation:
INFO:   Dice Score: 0.3896
INFO:   Average Inference Time per image: 0.0098 seconds
INFO:   Mean IoU (excluding NaN): 0.3892
INFO:   Per-Class IoU (first 10 classes):
INFO:     Class   0: IoU = 0.7427
INFO:     Class   1: IoU = 0.5299
INFO:     Class   2: IoU = 0.5935
INFO:     Class   3: IoU = 0.4436
INFO:     Class   4: IoU = 0.0243
INFO:     Class   5: IoU = 0.0010
INFO: Validation Dice score: 0.3895871639251709
INFO: Checkpoint 20 saved to results_ablation4_C_hybrid_sampler_s018/checkpoints/checkpoint_epoch20.pth!
wandb: Waiting for W&B process to finish... (success).
wandb:
wandb: Run history:
wandb:           epoch ▁▁▁▁▂▂▂▂▂▂▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▇▇▇▇▇▇████
wandb:   learning rate ███████████████▁▁▁▁▁
wandb:            step ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:      train loss ▄▅▂▄▄▃▃▃█▂▂▃▃▃▄▃▂▄▃▂▂▃▅▁▃▂▅▄▂▃▁▃▂▂▅▃▆▄▁▄
wandb: validation Dice ▁▇▁▄▇▄▅▇▇█▅▅▆▄████▅▄
wandb:
wandb: Run summary:
wandb:           epoch 20
wandb:   learning rate 1e-05
wandb:            step 2140
wandb:      train loss 0.49946
wandb: validation Dice 0.38959
wandb:
wandb: You can sync this run to the cloud by running:
wandb: wandb sync /workspace/SSCloud/train/wandb/offline-run-20260418_154039-3lw1zg5b
wandb: Find logs at: ./wandb/offline-run-20260418_154039-3lw1zg5b/logs

DONE: [C_Hybrid_sampler_s018]

Stage duration: 00:09:59

# D Same as reference with tighter max sample weight
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

Detailed metrics saved to results_ablation4_D_hybrid_sampler_cap15
  - Confusion matrix: results_ablation4_D_hybrid_sampler_cap15/confusion_matrix_epoch20.png
  - Per-class IoU: results_ablation4_D_hybrid_sampler_cap15/per_class_iou_epoch20.txt
  - Average inference time per image: 0.0092 seconds
INFO: Final Epoch 20 - Detailed Evaluation:
INFO:   Dice Score: 0.4624
INFO:   Average Inference Time per image: 0.0092 seconds
INFO:   Mean IoU (excluding NaN): 0.3924
INFO:   Per-Class IoU (first 10 classes):
INFO:     Class   0: IoU = 0.7418
INFO:     Class   1: IoU = 0.5528
INFO:     Class   2: IoU = 0.6126
INFO:     Class   3: IoU = 0.4358
INFO:     Class   4: IoU = 0.0117
INFO:     Class   5: IoU = 0.0000
INFO: Validation Dice score: 0.4623715877532959
INFO: Checkpoint 20 saved to results_ablation4_D_hybrid_sampler_cap15/checkpoints/checkpoint_epoch20.pth!
wandb: Waiting for W&B process to finish... (success).
wandb:
wandb: Run history:
wandb:           epoch ▁▁▁▁▂▂▂▂▂▂▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▇▇▇▇▇▇████
wandb:   learning rate █████████▂▂▂▂▂▂▁▁▁▁▁
wandb:            step ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:      train loss ▂▄▃▂▁▂▂▂▂▂▄▁▂▂▁▃▂▂▃▃▃▂█▃▅▂▁▁▁▂▂▂▆▃▂▃▁▂▂▂
wandb: validation Dice ▄▁▅█████▆██████▇▇▇▆▇
wandb:
wandb: Run summary:
wandb:           epoch 20
wandb:   learning rate 0.0
wandb:            step 2140
wandb:      train loss 1.40109
wandb: validation Dice 0.46237
wandb:
wandb: You can sync this run to the cloud by running:
wandb: wandb sync /workspace/SSCloud/train/wandb/offline-run-20260418_155041-vqr2nuwj
wandb: Find logs at: ./wandb/offline-run-20260418_155041-vqr2nuwj/logs

DONE: [D_Hybrid_sampler_cap15]

Stage duration: 00:09:46

============================================================
ALL EXPERIMENTS COMPLETED
Total elapsed time: 00:39:26
============================================================

Разобрал train/ablations/ablation_with_6_classes/summary/summary_run_ablation_4.md.
Итоги ablation 4:
- A_Hybrid_sampler_ref: Dice 0.5078, mIoU 0.3853, time 0.0104
- B_Hybrid_sampler_s012: Dice 0.4224, mIoU 0.3910, time 0.0093
- C_Hybrid_sampler_s018: Dice 0.3896, mIoU 0.3892, time 0.0098
- D_Hybrid_sampler_cap15: Dice 0.4624, mIoU 0.3924, time 0.0092
Что это значит:
- По Dice лучший в этой серии — A (0.5078).
- По mIoU лучший — D (0.3924), но Dice у него заметно ниже A.
- Усиление sampler (C, strength=0.18) дало небольшой плюс по редкому class 4 (0.0243), но сильно уронило Dice.
- В целом ablation 4 не побил лучший Dice из ablation 3 (B_Hybrid_sampler_plus = 0.5481), т.е. узкий тюнинг вокруг него оказался нестабильным.
Практический вывод:
- Текущий кандидат по глобальному качеству остается ablation 3 / B_Hybrid_sampler_plus.
- Для ablation 5 логично делать не новый широкий поиск, а confirm-run на 50 эпох для 2 конфигов:
  1) ablation3_B_hybrid_sampler_plus (кандидат по Dice)  
  2) ablation4_D_hybrid_sampler_cap15 (кандидат по mIoU/балансу)
Если хочешь, сразу напишу run_ablation_5.sh именно в этом формате (2 full-run по 50 эпох, memory-safe).