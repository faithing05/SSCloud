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

wandb: Waiting for W&B process to finish... (success).
wandb:
wandb: Run history:
wandb:           epoch ▁▁▁▁▂▂▂▂▃▃▃▃▃▃▃▃▄▄▄▄▅▅▅▅▆▆▆▆▆▆▆▆▇▇▇▇████
wandb:   learning rate ▁▁▁▁▁▁▁▁▁▁
wandb:            step ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:      train loss ▄▄▂▁▂█▂▂▂▄▂▂▂▁▂▂▄▂▂▂▁▃▁▅▂▂▃▂▁▁▂▁▁▂▁▂▁▂▁▂
wandb: validation Dice ▄▁▄▅█▂▆▆▅▆
wandb:
wandb: Run summary:
wandb:           epoch 10
wandb:   learning rate 0.0001
wandb:            step 1260
wandb:      train loss 0.70406
wandb: validation Dice 0.56723
wandb:
wandb: You can sync this run to the cloud by running:
wandb: wandb sync /workspace/SSCloud/train/wandb/offline-run-20260412_194843-28wyotee
wandb: Find logs at: ./wandb/offline-run-20260412_194843-28wyotee/logs

DONE: [A_Baseline]
Stage duration: 00:08:38

# B Class weights only
INFO: Starting training:
        Epochs:          10
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

wandb: Waiting for W&B process to finish... (success).
wandb:
wandb: Run history:
wandb:           epoch ▁▁▁▁▂▂▂▂▃▃▃▃▃▃▃▃▄▄▄▄▅▅▅▅▆▆▆▆▆▆▆▆▇▇▇▇████
wandb:   learning rate ▁▁▁▁▁▁▁▁▁▁
wandb:            step ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:      train loss ▅▄█▄▄▂▄▄▃▄▂▃▁▂▅▂▁▃▂▂▁▄▃▂▃▃▆▁▄▃▃▃▂▁▃▁▃▄▃▅
wandb: validation Dice ▄▇▅▁▇▇▅▇█▇
wandb:
wandb: Run summary:
wandb:           epoch 10
wandb:   learning rate 8e-05
wandb:            step 1260
wandb:      train loss 3.11316
wandb: validation Dice 0.50075
wandb:
wandb: You can sync this run to the cloud by running:
wandb: wandb sync /workspace/SSCloud/train/wandb/offline-run-20260412_195653-sw8r902a
wandb: Find logs at: ./wandb/offline-run-20260412_195653-sw8r902a/logs

DONE: [B_Weights]
Stage duration: 00:07:53

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

wandb: Waiting for W&B process to finish... (success).
wandb:
wandb: Run history:
wandb:           epoch ▁▁▁▁▂▂▂▂▃▃▃▃▃▃▃▃▄▄▄▄▅▅▅▅▆▆▆▆▆▆▆▆▇▇▇▇████
wandb:   learning rate ▁▁▁▁▁▁▁▁▁▁
wandb:            step ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:      train loss █▅▂▅▅▆▇▃▂▆▃▄▂▂▇▂▁▃▃▄▃▄▅▃▄▃▅▂▅▃▃▂▁▂▃▂▁▃▃▁
wandb: validation Dice ▆▂▃▁▃▂██▃▁
wandb:
wandb: Run summary:
wandb:           epoch 10
wandb:   learning rate 0.0001
wandb:            step 1260
wandb:      train loss 1.888
wandb: validation Dice 0.53886
wandb:
wandb: You can sync this run to the cloud by running:
wandb: wandb sync /workspace/SSCloud/train/wandb/offline-run-20260412_200437-mlb6f28j
wandb: Find logs at: ./wandb/offline-run-20260412_200437-mlb6f28j/logs

DONE: [C_Sampler]
Stage duration: 00:07:02        

# D Weights + oversampling
INFO: Starting training:
        Epochs:          10
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

wandb: Waiting for W&B process to finish... (success).
wandb:
wandb: Run history:
wandb:           epoch ▁▁▁▁▂▂▂▂▃▃▃▃▃▃▃▃▄▄▄▄▅▅▅▅▆▆▆▆▆▆▆▆▇▇▇▇████
wandb:   learning rate ████████▁▁
wandb:            step ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:      train loss █▅▁▂▄▃█▄▅▆▅▅▂▆▄▂▃▁▃▇▃▄▇█▃▃▅▃▃▃▁▄▁▆▂▅▃▄▂▆
wandb: validation Dice ▄▄▇▃▄▁▄▄▆█
wandb:
wandb: Run summary:
wandb:           epoch 10
wandb:   learning rate 1e-05
wandb:            step 1260
wandb:      train loss 3.56286
wandb: validation Dice 0.4163
wandb:
wandb: You can sync this run to the cloud by running:
wandb: wandb sync /workspace/SSCloud/train/wandb/offline-run-20260412_201137-9npd7lxp
wandb: Find logs at: ./wandb/offline-run-20260412_201137-9npd7lxp/logs

DONE: [D_Both]
Stage duration: 00:07:00

ALL EXPERIMENTS COMPLETED
Total elapsed time: 00:30:33

Посмотрел .md — отличный диагностический прогон, и он очень показательный.
Итог по качеству (лучше → хуже):
- A_Baseline: 0.56723 (лучший)
- C_Sampler: 0.53886
- B_Weights: 0.50075
- D_Both: 0.41630 (худший)
Что это значит:
- class weights в твоем датасете сейчас, скорее всего, вредят.
- oversampling отдельно еще терпимо, но хуже baseline.
- комбинация weights + oversampling явно перегружает обучение (в D еще и LR просел до 1e-05 к концу).
Что делать сейчас:
1. В боевой прогон (run_ablation_full.sh) запускать в приоритете A_Baseline и C_Sampler.
2. B и особенно D можно оставить только как контроль, либо временно исключить, чтобы не тратить GPU.
3. После full-прогона выбрать финальный конфиг по последним эпохам + стабильности кривой Dice.
Нюанс интерпретации:
- Validation size: 13 — валидация маленькая, разброс может быть заметный. Если будет время, лучший конфиг стоит перепроверить вторым запуском (другая random split seed), чтобы убедиться, что победа стабильна.