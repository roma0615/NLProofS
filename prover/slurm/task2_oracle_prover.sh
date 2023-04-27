/n/fs/484-nlproofs/miniconda3/envs/nlproofs/lib/python3.9/site-packages/ete3-3.1.2-py3.7.egg/ete3/evol/parser/codemlparser.py:221: SyntaxWarning: "is" with a literal. Did you mean "=="?
/n/fs/484-nlproofs/miniconda3/envs/nlproofs/lib/python3.9/site-packages/ete3-3.1.2-py3.7.egg/ete3/evol/parser/codemlparser.py:221: SyntaxWarning: "is" with a literal. Did you mean "=="?
Global seed set to 1
Some weights of the model checkpoint at roberta-large were not used when initializing RobertaModel: ['lm_head.layer_norm.weight', 'lm_head.dense.weight', 'lm_head.dense.bias', 'lm_head.layer_norm.bias', 'lm_head.decoder.weight', 'lm_head.bias']
- This IS expected if you are initializing RobertaModel from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing RobertaModel from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
/n/fs/484-nlproofs/miniconda3/envs/nlproofs/lib/python3.9/site-packages/torchmetrics/utilities/prints.py:36: UserWarning: Metric `AveragePrecision` will save all targets and predictions in buffer. For large datasets this may lead to large memory footprint.
  warnings.warn(*args, **kwargs)
Multiprocessing is handled by SLURM.
GPU available: True, used: True
TPU available: False, using: 0 TPU cores
IPU available: False, using: 0 IPUs
HPU available: False, using: 0 HPUs
Restoring states from the checkpoint path at /n/fs/484-nlproofs/weights/task2_prover.ckpt
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Loaded model weights from checkpoint at /n/fs/484-nlproofs/weights/task2_prover.ckpt
/n/fs/484-nlproofs/miniconda3/envs/nlproofs/lib/python3.9/site-packages/pytorch_lightning/trainer/connectors/data_connector.py:240: PossibleUserWarning: The dataloader, val_dataloader 0, does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` (try 40 which is the number of cpus on this machine) in the `DataLoader` init to improve performance.
  rank_zero_warn(
187 proofs loaded. 0 invalid ones removed.
Validation: 0it [00:00, ?it/s]Validation:   0%|          | 0/94 [00:00<?, ?it/s]Validation DataLoader 0:   0%|          | 0/94 [00:00<?, ?it/s]Validation DataLoader 0:   1%|          | 1/94 [00:32<50:19, 32.47s/it]Validation DataLoader 0:   1%|          | 1/94 [00:32<50:19, 32.47s/it]Validation DataLoader 0:   2%|▏         | 2/94 [00:55<45:17, 29.54s/it]Validation DataLoader 0:   2%|▏         | 2/94 [00:55<45:17, 29.54s/it]Validation DataLoader 0:   3%|▎         | 3/94 [01:29<46:56, 30.95s/it]Validation DataLoader 0:   3%|▎         | 3/94 [01:29<46:56, 30.95s/it]Validation DataLoader 0:   4%|▍         | 4/94 [01:45<39:50, 26.56s/it]Validation DataLoader 0:   4%|▍         | 4/94 [01:45<39:50, 26.56s/it]Validation DataLoader 0:   5%|▌         | 5/94 [02:21<43:32, 29.35s/it]Validation DataLoader 0:   5%|▌         | 5/94 [02:21<43:32, 29.35s/it]Validation DataLoader 0:   6%|▋         | 6/94 [02:35<36:08, 24.64s/it]Validation DataLoader 0:   6%|▋         | 6/94 [02:35<36:08, 24.64s/it]Validation DataLoader 0:   7%|▋         | 7/94 [02:53<32:52, 22.67s/it]Validation DataLoader 0:   7%|▋         | 7/94 [02:53<32:52, 22.67s/it]Validation DataLoader 0:   9%|▊         | 8/94 [03:02<26:38, 18.58s/it]Validation DataLoader 0:   9%|▊         | 8/94 [03:02<26:38, 18.58s/it]Validation DataLoader 0:  10%|▉         | 9/94 [03:13<22:57, 16.21s/it]Validation DataLoader 0:  10%|▉         | 9/94 [03:13<22:57, 16.21s/it]Validation DataLoader 0:  11%|█         | 10/94 [04:27<46:59, 33.56s/it]Validation DataLoader 0:  11%|█         | 10/94 [04:27<46:59, 33.56s/it]Validation DataLoader 0:  12%|█▏        | 11/94 [04:43<39:30, 28.56s/it]Validation DataLoader 0:  12%|█▏        | 11/94 [04:43<39:30, 28.56s/it]Validation DataLoader 0:  13%|█▎        | 12/94 [05:06<36:24, 26.64s/it]Validation DataLoader 0:  13%|█▎        | 12/94 [05:06<36:24, 26.64s/it]Validation DataLoader 0:  14%|█▍        | 13/94 [05:32<35:52, 26.57s/it]Validation DataLoader 0:  14%|█▍        | 13/94 [05:32<35:52, 26.57s/it]Validation DataLoader 0:  15%|█▍        | 14/94 [06:16<42:21, 31.77s/it]Validation DataLoader 0:  15%|█▍        | 14/94 [06:16<42:21, 31.77s/it]Validation DataLoader 0:  16%|█▌        | 15/94 [07:01<47:07, 35.79s/it]Validation DataLoader 0:  16%|█▌        | 15/94 [07:01<47:07, 35.79s/it]Validation DataLoader 0:  17%|█▋        | 16/94 [08:04<57:16, 44.05s/it]Validation DataLoader 0:  17%|█▋        | 16/94 [08:04<57:16, 44.05s/it]Validation DataLoader 0:  18%|█▊        | 17/94 [09:07<1:03:40, 49.61s/it]Validation DataLoader 0:  18%|█▊        | 17/94 [09:07<1:03:40, 49.61s/it]Validation DataLoader 0:  19%|█▉        | 18/94 [09:52<1:00:55, 48.09s/it]Validation DataLoader 0:  19%|█▉        | 18/94 [09:52<1:00:55, 48.09s/it]Validation DataLoader 0:  20%|██        | 19/94 [10:10<48:51, 39.08s/it]  Validation DataLoader 0:  20%|██        | 19/94 [10:10<48:51, 39.08s/it]Validation DataLoader 0:  21%|██▏       | 20/94 [10:40<45:05, 36.56s/it]Validation DataLoader 0:  21%|██▏       | 20/94 [10:40<45:05, 36.56s/it]Validation DataLoader 0:  22%|██▏       | 21/94 [11:30<49:16, 40.50s/it]Validation DataLoader 0:  22%|██▏       | 21/94 [11:30<49:16, 40.50s/it]Validation DataLoader 0:  23%|██▎       | 22/94 [12:31<56:07, 46.77s/it]Validation DataLoader 0:  23%|██▎       | 22/94 [12:31<56:07, 46.77s/it]Validation DataLoader 0:  24%|██▍       | 23/94 [13:34<1:01:05, 51.63s/it]Validation DataLoader 0:  24%|██▍       | 23/94 [13:34<1:01:05, 51.63s/it]Validation DataLoader 0:  26%|██▌       | 24/94 [14:02<51:43, 44.33s/it]  Validation DataLoader 0:  26%|██▌       | 24/94 [14:02<51:43, 44.33s/it]Validation DataLoader 0:  27%|██▋       | 25/94 [14:28<44:47, 38.96s/it]Validation DataLoader 0:  27%|██▋       | 25/94 [14:28<44:47, 38.96s/it]Validation DataLoader 0:  28%|██▊       | 26/94 [14:49<37:54, 33.45s/it]Validation DataLoader 0:  28%|██▊       | 26/94 [14:49<37:54, 33.45s/it]Validation DataLoader 0:  29%|██▊       | 27/94 [15:30<39:52, 35.71s/it]Validation DataLoader 0:  29%|██▊       | 27/94 [15:30<39:52, 35.71s/it]Validation DataLoader 0:  30%|██▉       | 28/94 [15:52<34:48, 31.64s/it]Validation DataLoader 0:  30%|██▉       | 28/94 [15:52<34:48, 31.64s/it]Validation DataLoader 0:  31%|███       | 29/94 [16:02<27:08, 25.06s/it]Validation DataLoader 0:  31%|███       | 29/94 [16:02<27:08, 25.06s/it]Validation DataLoader 0:  32%|███▏      | 30/94 [16:42<31:45, 29.77s/it]Validation DataLoader 0:  32%|███▏      | 30/94 [16:42<31:45, 29.77s/it]Validation DataLoader 0:  33%|███▎      | 31/94 [17:07<29:40, 28.27s/it]Validation DataLoader 0:  33%|███▎      | 31/94 [17:07<29:40, 28.27s/it]Validation DataLoader 0:  34%|███▍      | 32/94 [18:08<39:11, 37.93s/it]Validation DataLoader 0:  34%|███▍      | 32/94 [18:08<39:11, 37.93s/it]Validation DataLoader 0:  35%|███▌      | 33/94 [18:21<30:59, 30.49s/it]Validation DataLoader 0:  35%|███▌      | 33/94 [18:21<30:59, 30.49s/it]Validation DataLoader 0:  36%|███▌      | 34/94 [20:25<58:36, 58.61s/it]Validation DataLoader 0:  36%|███▌      | 34/94 [20:25<58:36, 58.61s/it]Validation DataLoader 0:  37%|███▋      | 35/94 [20:38<44:14, 44.99s/it]Validation DataLoader 0:  37%|███▋      | 35/94 [20:38<44:14, 44.99s/it]Validation DataLoader 0:  38%|███▊      | 36/94 [20:50<33:46, 34.94s/it]Validation DataLoader 0:  38%|███▊      | 36/94 [20:50<33:46, 34.94s/it]Validation DataLoader 0:  39%|███▉      | 37/94 [21:09<28:53, 30.42s/it]Validation DataLoader 0:  39%|███▉      | 37/94 [21:09<28:53, 30.42s/it]Validation DataLoader 0:  40%|████      | 38/94 [22:18<39:00, 41.79s/it]Validation DataLoader 0:  40%|████      | 38/94 [22:18<39:00, 41.79s/it]Validation DataLoader 0:  41%|████▏     | 39/94 [23:17<43:11, 47.11s/it]Validation DataLoader 0:  41%|████▏     | 39/94 [23:17<43:11, 47.11s/it]Validation DataLoader 0:  43%|████▎     | 40/94 [23:45<37:08, 41.26s/it]Validation DataLoader 0:  43%|████▎     | 40/94 [23:45<37:08, 41.26s/it]Validation DataLoader 0:  44%|████▎     | 41/94 [24:18<34:15, 38.78s/it]Validation DataLoader 0:  44%|████▎     | 41/94 [24:18<34:15, 38.78s/it]Validation DataLoader 0:  45%|████▍     | 42/94 [24:28<26:13, 30.25s/it]Validation DataLoader 0:  45%|████▍     | 42/94 [24:28<26:13, 30.25s/it]Validation DataLoader 0:  46%|████▌     | 43/94 [25:02<26:30, 31.18s/it]Validation DataLoader 0:  46%|████▌     | 43/94 [25:02<26:30, 31.18s/it]Validation DataLoader 0:  47%|████▋     | 44/94 [27:11<50:32, 60.66s/it]Validation DataLoader 0:  47%|████▋     | 44/94 [27:11<50:32, 60.66s/it]Validation DataLoader 0:  48%|████▊     | 45/94 [28:22<52:06, 63.80s/it]Validation DataLoader 0:  48%|████▊     | 45/94 [28:22<52:06, 63.80s/it]Validation DataLoader 0:  49%|████▉     | 46/94 [30:03<59:54, 74.88s/it]Validation DataLoader 0:  49%|████▉     | 46/94 [30:03<59:54, 74.88s/it]Validation DataLoader 0:  50%|█████     | 47/94 [31:24<1:00:13, 76.89s/it]Validation DataLoader 0:  50%|█████     | 47/94 [31:24<1:00:13, 76.89s/it]Validation DataLoader 0:  51%|█████     | 48/94 [32:47<1:00:14, 78.57s/it]Validation DataLoader 0:  51%|█████     | 48/94 [32:47<1:00:14, 78.57s/it]Validation DataLoader 0:  52%|█████▏    | 49/94 [33:04<45:00, 60.01s/it]  Validation DataLoader 0:  52%|█████▏    | 49/94 [33:04<45:00, 60.01s/it]Validation DataLoader 0:  53%|█████▎    | 50/94 [34:28<49:16, 67.18s/it]Validation DataLoader 0:  53%|█████▎    | 50/94 [34:28<49:16, 67.18s/it]Validation DataLoader 0:  54%|█████▍    | 51/94 [36:00<53:32, 74.71s/it]Validation DataLoader 0:  54%|█████▍    | 51/94 [36:00<53:32, 74.71s/it]Validation DataLoader 0:  55%|█████▌    | 52/94 [37:05<50:11, 71.70s/it]Validation DataLoader 0:  55%|█████▌    | 52/94 [37:05<50:11, 71.70s/it]Validation DataLoader 0:  56%|█████▋    | 53/94 [37:32<39:55, 58.43s/it]Validation DataLoader 0:  56%|█████▋    | 53/94 [37:32<39:55, 58.43s/it]Validation DataLoader 0:  57%|█████▋    | 54/94 [37:46<30:09, 45.23s/it]Validation DataLoader 0:  57%|█████▋    | 54/94 [37:46<30:09, 45.23s/it]Validation DataLoader 0:  59%|█████▊    | 55/94 [38:07<24:30, 37.71s/it]Validation DataLoader 0:  59%|█████▊    | 55/94 [38:07<24:30, 37.71s/it]Validation DataLoader 0:  60%|█████▉    | 56/94 [38:31<21:25, 33.83s/it]Validation DataLoader 0:  60%|█████▉    | 56/94 [38:31<21:25, 33.83s/it]Validation DataLoader 0:  61%|██████    | 57/94 [39:04<20:42, 33.59s/it]Validation DataLoader 0:  61%|██████    | 57/94 [39:04<20:42, 33.59s/it]Validation DataLoader 0:  62%|██████▏   | 58/94 [39:39<20:16, 33.79s/it]Validation DataLoader 0:  62%|██████▏   | 58/94 [39:39<20:16, 33.79s/it]Validation DataLoader 0:  63%|██████▎   | 59/94 [39:50<15:47, 27.07s/it]Validation DataLoader 0:  63%|██████▎   | 59/94 [39:50<15:47, 27.07s/it]Validation DataLoader 0:  64%|██████▍   | 60/94 [43:40<49:45, 87.81s/it]Validation DataLoader 0:  64%|██████▍   | 60/94 [43:40<49:45, 87.81s/it]Validation DataLoader 0:  65%|██████▍   | 61/94 [44:31<42:21, 77.01s/it]Validation DataLoader 0:  65%|██████▍   | 61/94 [44:31<42:21, 77.01s/it]Validation DataLoader 0:  66%|██████▌   | 62/94 [44:59<33:05, 62.05s/it]Validation DataLoader 0:  66%|██████▌   | 62/94 [44:59<33:05, 62.05s/it]Validation DataLoader 0:  67%|██████▋   | 63/94 [45:19<25:38, 49.62s/it]Validation DataLoader 0:  67%|██████▋   | 63/94 [45:19<25:38, 49.62s/it]Validation DataLoader 0:  68%|██████▊   | 64/94 [47:49<39:52, 79.77s/it]Validation DataLoader 0:  68%|██████▊   | 64/94 [47:49<39:52, 79.77s/it]Validation DataLoader 0:  69%|██████▉   | 65/94 [50:08<47:03, 97.37s/it]Validation DataLoader 0:  69%|██████▉   | 65/94 [50:08<47:03, 97.37s/it]Validation DataLoader 0:  70%|███████   | 66/94 [50:36<35:48, 76.75s/it]Validation DataLoader 0:  70%|███████   | 66/94 [50:36<35:48, 76.75s/it]Validation DataLoader 0:  71%|███████▏  | 67/94 [50:56<26:46, 59.51s/it]Validation DataLoader 0:  71%|███████▏  | 67/94 [50:56<26:46, 59.51s/it]Validation DataLoader 0:  72%|███████▏  | 68/94 [51:06<19:24, 44.80s/it]Validation DataLoader 0:  72%|███████▏  | 68/94 [51:06<19:24, 44.80s/it]Validation DataLoader 0:  73%|███████▎  | 69/94 [51:25<15:28, 37.15s/it]Validation DataLoader 0:  73%|███████▎  | 69/94 [51:25<15:28, 37.15s/it]Validation DataLoader 0:  74%|███████▍  | 70/94 [51:41<12:19, 30.82s/it]Validation DataLoader 0:  74%|███████▍  | 70/94 [51:41<12:19, 30.82s/it]Validation DataLoader 0:  76%|███████▌  | 71/94 [52:03<10:48, 28.18s/it]Validation DataLoader 0:  76%|███████▌  | 71/94 [52:03<10:48, 28.18s/it]Validation DataLoader 0:  77%|███████▋  | 72/94 [52:34<10:34, 28.85s/it]Validation DataLoader 0:  77%|███████▋  | 72/94 [52:34<10:34, 28.85s/it]Validation DataLoader 0:  78%|███████▊  | 73/94 [52:50<08:47, 25.10s/it]Validation DataLoader 0:  78%|███████▊  | 73/94 [52:50<08:47, 25.10s/it]Validation DataLoader 0:  79%|███████▊  | 74/94 [53:16<08:23, 25.16s/it]Validation DataLoader 0:  79%|███████▊  | 74/94 [53:16<08:23, 25.16s/it]Validation DataLoader 0:  80%|███████▉  | 75/94 [53:50<08:49, 27.85s/it]Validation DataLoader 0:  80%|███████▉  | 75/94 [53:50<08:49, 27.85s/it]Validation DataLoader 0:  81%|████████  | 76/94 [54:31<09:35, 31.99s/it]Validation DataLoader 0:  81%|████████  | 76/94 [54:31<09:35, 31.99s/it]Validation DataLoader 0:  82%|████████▏ | 77/94 [54:42<07:16, 25.70s/it]Validation DataLoader 0:  82%|████████▏ | 77/94 [54:42<07:16, 25.70s/it]Validation DataLoader 0:  83%|████████▎ | 78/94 [55:04<06:29, 24.35s/it]Validation DataLoader 0:  83%|████████▎ | 78/94 [55:04<06:29, 24.35s/it]Validation DataLoader 0:  84%|████████▍ | 79/94 [55:17<05:14, 20.94s/it]Validation DataLoader 0:  84%|████████▍ | 79/94 [55:17<05:14, 20.94s/it]Validation DataLoader 0:  85%|████████▌ | 80/94 [55:45<05:26, 23.30s/it]Validation DataLoader 0:  85%|████████▌ | 80/94 [55:45<05:26, 23.30s/it]Validation DataLoader 0:  86%|████████▌ | 81/94 [57:06<08:48, 40.64s/it]Validation DataLoader 0:  86%|████████▌ | 81/94 [57:06<08:48, 40.64s/it]Validation DataLoader 0:  87%|████████▋ | 82/94 [57:34<07:21, 36.76s/it]Validation DataLoader 0:  87%|████████▋ | 82/94 [57:34<07:21, 36.76s/it]Validation DataLoader 0:  88%|████████▊ | 83/94 [57:56<05:55, 32.34s/it]Validation DataLoader 0:  88%|████████▊ | 83/94 [57:56<05:55, 32.34s/it]Validation DataLoader 0:  89%|████████▉ | 84/94 [58:12<04:34, 27.46s/it]Validation DataLoader 0:  89%|████████▉ | 84/94 [58:12<04:34, 27.46s/it]Validation DataLoader 0:  90%|█████████ | 85/94 [58:26<03:29, 23.24s/it]Validation DataLoader 0:  90%|█████████ | 85/94 [58:26<03:29, 23.24s/it]Validation DataLoader 0:  91%|█████████▏| 86/94 [59:08<03:50, 28.83s/it]Validation DataLoader 0:  91%|█████████▏| 86/94 [59:08<03:50, 28.83s/it]Validation DataLoader 0:  93%|█████████▎| 87/94 [59:48<03:46, 32.43s/it]Validation DataLoader 0:  93%|█████████▎| 87/94 [59:48<03:46, 32.43s/it]Validation DataLoader 0:  94%|█████████▎| 88/94 [59:58<02:34, 25.74s/it]Validation DataLoader 0:  94%|█████████▎| 88/94 [59:58<02:34, 25.74s/it]Validation DataLoader 0:  95%|█████████▍| 89/94 [1:00:20<02:01, 24.34s/it]Validation DataLoader 0:  95%|█████████▍| 89/94 [1:00:20<02:01, 24.34s/it]Validation DataLoader 0:  96%|█████████▌| 90/94 [1:00:35<01:26, 21.57s/it]Validation DataLoader 0:  96%|█████████▌| 90/94 [1:00:35<01:26, 21.57s/it]Validation DataLoader 0:  97%|█████████▋| 91/94 [1:00:47<00:56, 18.73s/it]Validation DataLoader 0:  97%|█████████▋| 91/94 [1:00:47<00:56, 18.73s/it]Validation DataLoader 0:  98%|█████████▊| 92/94 [1:01:17<00:44, 22.31s/it]Validation DataLoader 0:  98%|█████████▊| 92/94 [1:01:17<00:44, 22.31s/it]Validation DataLoader 0:  99%|█████████▉| 93/94 [1:01:44<00:23, 23.48s/it]Validation DataLoader 0:  99%|█████████▉| 93/94 [1:01:44<00:23, 23.48s/it]Validation DataLoader 0: 100%|██████████| 94/94 [1:01:49<00:00, 18.13s/it]Validation DataLoader 0: 100%|██████████| 94/94 [1:01:49<00:00, 18.13s/it]  0%|          | 0/187 [00:00<?, ?it/s]100%|██████████| 187/187 [00:00<00:00, 2134.98it/s]
Validation results saved to /n/fs/484-nlproofs/zs0608/NLProofS/prover/lightning_logs/version_18906425/results_val.json and /n/fs/484-nlproofs/zs0608/NLProofS/prover/lightning_logs/version_18906425/results_val.tsv
Performance by depth:
51 trees have depth 1
	leaves: 1.0	1.0
	steps: 1.0	1.0
	proof: 1.0	1.0
64 trees have depth 2
	leaves: 0.8125	0.9645957341269842
	steps: 0.765625	0.811235119047619
	proof: 0.765625	0.765625
28 trees have depth 3
	leaves: 0.7142857142857143	0.9440953887382458
	steps: 0.5	0.6640306122448979
	proof: 0.5	0.5
18 trees have depth 4
	leaves: 0.7222222222222222	0.9367424242424244
	steps: 0.5555555555555556	0.6769079685746352
	proof: 0.5555555555555556	0.5555555555555556
13 trees have depth 5
	leaves: 0.38461538461538464	0.8729238035918198
	steps: 0.23076923076923078	0.5143020378314496
	proof: 0.23076923076923078	0.23076923076923078
7 trees have depth 6
	leaves: 0.2857142857142857	0.716122035858878
	steps: 0.2857142857142857	0.4242825801649331
	proof: 0.2857142857142857	0.2857142857142857
6 trees have depth 7
	leaves: 0.3333333333333333	0.7947241291634884
	steps: 0.16666666666666666	0.4304014939309057
	proof: 0.16666666666666666	0.16666666666666666
Performance by size:
41 trees have size 3
	leaves: 1.0	1.0
	steps: 1.0	1.0
	proof: 1.0	1.0
10 trees have size 4
	leaves: 1.0	1.0
	steps: 1.0	1.0
	proof: 1.0	1.0
37 trees have size 5
	leaves: 0.8378378378378378	0.9639639639639639
	steps: 0.8108108108108109	0.8378378378378378
	proof: 0.8108108108108109	0.8108108108108109
13 trees have size 6
	leaves: 0.7692307692307693	0.9670329670329672
	steps: 0.6923076923076923	0.7615384615384616
	proof: 0.6923076923076923	0.6923076923076923
26 trees have size 7
	leaves: 0.7307692307692307	0.950091575091575
	steps: 0.6153846153846154	0.72014652014652
	proof: 0.6153846153846154	0.6153846153846154
5 trees have size 8
	leaves: 1.0	1.0
	steps: 0.6	0.7333333333333333
	proof: 0.6	0.6
10 trees have size 9
	leaves: 0.7	0.945909090909091
	steps: 0.5	0.6526984126984127
	proof: 0.5	0.5
2 trees have size 10
	leaves: 0.5	0.9545454545454546
	steps: 0.5	0.625
	proof: 0.5	0.5
14 trees have size 11
	leaves: 0.7142857142857143	0.9311355311355312
	steps: 0.7142857142857143	0.7882653061224489
	proof: 0.7142857142857143	0.7142857142857143
1 trees have size 12
	leaves: 1.0	1.0
	steps: 0.0	0.6666666666666665
	proof: 0.0	0.0
4 trees have size 13
	leaves: 0.5	0.8741258741258742
	steps: 0.0	0.3484848484848484
	proof: 0.0	0.0
1 trees have size 15
	leaves: 0.0	0.4
	steps: 0.0	0.25
	proof: 0.0	0.0
2 trees have size 16
	leaves: 0.5	0.9375
	steps: 0.5	0.5769230769230769
	proof: 0.5	0.5
6 trees have size 17
	leaves: 0.16666666666666666	0.6799201529464688
	steps: 0.0	0.23842627960275017
	proof: 0.0	0.0
2 trees have size 18
	leaves: 0.0	0.8750000000000001
	steps: 0.0	0.32142857142857145
	proof: 0.0	0.0
3 trees have size 19
	leaves: 0.6666666666666666	0.9824561403508772
	steps: 0.6666666666666666	0.8571428571428572
	proof: 0.6666666666666666	0.6666666666666666
1 trees have size 20
	leaves: 1.0	1.0
	steps: 1.0	1.0
	proof: 1.0	1.0
3 trees have size 21
	leaves: 0.3333333333333333	0.891025641025641
	steps: 0.0	0.5015873015873016
	proof: 0.0	0.0
3 trees have size 23
	leaves: 0.6666666666666666	0.9444444444444445
	steps: 0.3333333333333333	0.668148148148148
	proof: 0.3333333333333333	0.3333333333333333
2 trees have size 27
	leaves: 0.0	0.575657894736842
	steps: 0.0	0.058823529411764705
	proof: 0.0	0.0
1 trees have size 31
	leaves: 0.0	0.608695652173913
	steps: 0.0	0.09523809523809522
	proof: 0.0	0.0
Validation DataLoader 0: 100%|██████████| 94/94 [1:01:49<00:00, 39.47s/it]
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
     Validate metric           DataLoader 0
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  ExactMatch_leaves_val     0.7754010695187166
  ExactMatch_proof_val      0.6951871657754011
  ExactMatch_steps_val      0.6951871657754011
      F1_leaves_val         0.9473763100419693
      F1_proof_val          0.6951871657754011
      F1_steps_val          0.7803987257368824
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Configuration: 
 Namespace(config=None, subcommand='validate', validate=Namespace(config=[Path_fsr(cli_task2_stepwise_t5-large.yaml, cwd=/n/fs/484-nlproofs/zs0608/NLProofS/prover)], seed_everything=1, trainer=Namespace(logger=True, checkpoint_callback=None, enable_checkpointing=True, callbacks=[Namespace(class_path='pytorch_lightning.callbacks.LearningRateMonitor', init_args=Namespace(logging_interval='step', log_momentum=False))], default_root_dir=None, gradient_clip_val=0.5, gradient_clip_algorithm=None, process_position=0, num_nodes=1, num_processes=None, devices=None, gpus=1, auto_select_gpus=False, tpu_cores=None, ipus=None, log_gpu_memory=None, progress_bar_refresh_rate=None, enable_progress_bar=True, overfit_batches=0.0, track_grad_norm=-1, check_val_every_n_epoch=10, fast_dev_run=False, accumulate_grad_batches=32, max_epochs=600, min_epochs=None, max_steps=-1, min_steps=None, max_time=None, limit_train_batches=None, limit_val_batches=None, limit_test_batches=None, limit_predict_batches=None, val_check_interval=None, flush_logs_every_n_steps=None, log_every_n_steps=5, accelerator=None, strategy=None, sync_batchnorm=False, precision=32, enable_model_summary=True, weights_summary='top', weights_save_path=None, num_sanity_val_steps=2, resume_from_checkpoint=None, profiler=None, benchmark=None, deterministic=False, reload_dataloaders_every_n_epochs=0, auto_lr_find=False, replace_sampler_ddp=True, detect_anomaly=False, auto_scale_batch_size=False, prepare_data_per_node=None, plugins=None, amp_backend='native', amp_level=None, move_metrics_to_cpu=False, multiple_trainloader_mode='max_size_cycle', stochastic_weight_avg=False, terminate_on_nan=None), model=Namespace(stepwise=True, max_num_steps=20, model_name='t5-large', lr=5e-05, warmup_steps=1000, num_beams=10, topk=10, proof_search=True, verifier_weight=0.5, verifier_ckpt='/n/fs/484-nlproofs/weights/task2_verifier.ckpt', oracle_prover=True, oracle_verifier=False, dataset='entailmentbank', max_input_len=1024), data=Namespace(dataset='entailmentbank', sample_goal='intermediates', max_input_len=1024, max_output_len=64, batch_size=2, num_workers=2, path_train='../data/entailment_trees_emnlp2021_data_v3/dataset/task_2/train.jsonl', path_val='../data/entailment_trees_emnlp2021_data_v3/dataset/task_2/dev.jsonl', path_test='../data/entailment_trees_emnlp2021_data_v3/dataset/task_2/test.jsonl', subtree_proved_prob=0.75, subtree_proved_all_or_none=False, model_name='t5-large', stepwise=True), ckpt_path='/n/fs/484-nlproofs/weights/task2_prover.ckpt', verbose=True))
