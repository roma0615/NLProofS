/n/fs/484-nlproofs/miniconda3/envs/nlproofs/lib/python3.9/site-packages/ete3-3.1.2-py3.7.egg/ete3/evol/parser/codemlparser.py:221: SyntaxWarning: "is" with a literal. Did you mean "=="?
/n/fs/484-nlproofs/miniconda3/envs/nlproofs/lib/python3.9/site-packages/ete3-3.1.2-py3.7.egg/ete3/evol/parser/codemlparser.py:221: SyntaxWarning: "is" with a literal. Did you mean "=="?
Global seed set to 1
Some weights of the model checkpoint at roberta-large were not used when initializing RobertaModel: ['lm_head.decoder.weight', 'lm_head.dense.bias', 'lm_head.layer_norm.bias', 'lm_head.dense.weight', 'lm_head.layer_norm.weight', 'lm_head.bias']
- This IS expected if you are initializing RobertaModel from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing RobertaModel from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
/n/fs/484-nlproofs/miniconda3/envs/nlproofs/lib/python3.9/site-packages/torchmetrics/utilities/prints.py:36: UserWarning: Metric `AveragePrecision` will save all targets and predictions in buffer. For large datasets this may lead to large memory footprint.
  warnings.warn(*args, **kwargs)
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
Validation: 0it [00:00, ?it/s]Validation:   0%|          | 0/94 [00:00<?, ?it/s]Validation DataLoader 0:   0%|          | 0/94 [00:00<?, ?it/s]Validation DataLoader 0:   1%|          | 1/94 [00:52<1:22:02, 52.93s/it]Validation DataLoader 0:   1%|          | 1/94 [00:52<1:22:02, 52.93s/it]Validation DataLoader 0:   2%|▏         | 2/94 [01:09<48:06, 31.37s/it]  Validation DataLoader 0:   2%|▏         | 2/94 [01:09<48:06, 31.37s/it]Validation DataLoader 0:   3%|▎         | 3/94 [01:30<40:42, 26.84s/it]Validation DataLoader 0:   3%|▎         | 3/94 [01:30<40:42, 26.84s/it]Validation DataLoader 0:   4%|▍         | 4/94 [01:42<31:23, 20.93s/it]Validation DataLoader 0:   4%|▍         | 4/94 [01:42<31:23, 20.93s/it]Validation DataLoader 0:   5%|▌         | 5/94 [02:05<32:18, 21.78s/it]Validation DataLoader 0:   5%|▌         | 5/94 [02:05<32:18, 21.78s/it]Validation DataLoader 0:   6%|▋         | 6/94 [02:17<26:49, 18.29s/it]Validation DataLoader 0:   6%|▋         | 6/94 [02:17<26:49, 18.29s/it]Validation DataLoader 0:   7%|▋         | 7/94 [02:31<24:27, 16.87s/it]Validation DataLoader 0:   7%|▋         | 7/94 [02:31<24:27, 16.87s/it]Validation DataLoader 0:   9%|▊         | 8/94 [02:40<20:28, 14.29s/it]Validation DataLoader 0:   9%|▊         | 8/94 [02:40<20:28, 14.29s/it]Validation DataLoader 0:  10%|▉         | 9/94 [02:50<18:30, 13.07s/it]Validation DataLoader 0:  10%|▉         | 9/94 [02:50<18:30, 13.07s/it]Validation DataLoader 0:  11%|█         | 10/94 [03:37<32:55, 23.52s/it]Validation DataLoader 0:  11%|█         | 10/94 [03:37<32:55, 23.52s/it]Validation DataLoader 0:  12%|█▏        | 11/94 [03:49<27:40, 20.00s/it]Validation DataLoader 0:  12%|█▏        | 11/94 [03:49<27:40, 20.00s/it]Validation DataLoader 0:  13%|█▎        | 12/94 [04:06<26:00, 19.04s/it]Validation DataLoader 0:  13%|█▎        | 12/94 [04:06<26:00, 19.04s/it]Validation DataLoader 0:  14%|█▍        | 13/94 [04:34<29:35, 21.92s/it]Validation DataLoader 0:  14%|█▍        | 13/94 [04:34<29:35, 21.92s/it]Validation DataLoader 0:  15%|█▍        | 14/94 [05:34<44:21, 33.27s/it]Validation DataLoader 0:  15%|█▍        | 14/94 [05:34<44:21, 33.27s/it]Validation DataLoader 0:  16%|█▌        | 15/94 [06:22<49:54, 37.91s/it]Validation DataLoader 0:  16%|█▌        | 15/94 [06:22<49:54, 37.91s/it]Validation DataLoader 0:  17%|█▋        | 16/94 [07:07<51:54, 39.93s/it]Validation DataLoader 0:  17%|█▋        | 16/94 [07:07<51:54, 39.93s/it]Validation DataLoader 0:  18%|█▊        | 17/94 [07:33<45:41, 35.60s/it]Validation DataLoader 0:  18%|█▊        | 17/94 [07:33<45:41, 35.60s/it]Validation DataLoader 0:  19%|█▉        | 18/94 [08:04<43:22, 34.25s/it]Validation DataLoader 0:  19%|█▉        | 18/94 [08:04<43:22, 34.25s/it]Validation DataLoader 0:  20%|██        | 19/94 [08:17<34:58, 27.99s/it]Validation DataLoader 0:  20%|██        | 19/94 [08:17<34:58, 27.99s/it]Validation DataLoader 0:  21%|██▏       | 20/94 [08:50<36:12, 29.36s/it]Validation DataLoader 0:  21%|██▏       | 20/94 [08:50<36:12, 29.36s/it]Validation DataLoader 0:  22%|██▏       | 21/94 [09:54<48:38, 39.98s/it]Validation DataLoader 0:  22%|██▏       | 21/94 [09:54<48:38, 39.98s/it]Validation DataLoader 0:  23%|██▎       | 22/94 [10:29<45:58, 38.32s/it]Validation DataLoader 0:  23%|██▎       | 22/94 [10:29<45:58, 38.32s/it]Validation DataLoader 0:  24%|██▍       | 23/94 [10:58<41:59, 35.48s/it]Validation DataLoader 0:  24%|██▍       | 23/94 [10:58<41:59, 35.48s/it]Validation DataLoader 0:  26%|██▌       | 24/94 [11:25<38:29, 33.00s/it]Validation DataLoader 0:  26%|██▌       | 24/94 [11:25<38:29, 33.00s/it]Validation DataLoader 0:  27%|██▋       | 25/94 [11:42<32:24, 28.19s/it]Validation DataLoader 0:  27%|██▋       | 25/94 [11:42<32:24, 28.19s/it]Validation DataLoader 0:  28%|██▊       | 26/94 [11:57<27:36, 24.36s/it]Validation DataLoader 0:  28%|██▊       | 26/94 [11:57<27:36, 24.36s/it]Validation DataLoader 0:  29%|██▊       | 27/94 [12:40<33:22, 29.89s/it]Validation DataLoader 0:  29%|██▊       | 27/94 [12:40<33:22, 29.89s/it]Validation DataLoader 0:  30%|██▉       | 28/94 [12:57<28:32, 25.95s/it]Validation DataLoader 0:  30%|██▉       | 28/94 [12:57<28:32, 25.95s/it]Validation DataLoader 0:  31%|███       | 29/94 [13:06<22:45, 21.01s/it]Validation DataLoader 0:  31%|███       | 29/94 [13:06<22:45, 21.01s/it]Validation DataLoader 0:  32%|███▏      | 30/94 [13:39<26:04, 24.45s/it]Validation DataLoader 0:  32%|███▏      | 30/94 [13:39<26:04, 24.45s/it]Validation DataLoader 0:  33%|███▎      | 31/94 [14:06<26:37, 25.36s/it]Validation DataLoader 0:  33%|███▎      | 31/94 [14:06<26:37, 25.36s/it]Validation DataLoader 0:  34%|███▍      | 32/94 [14:31<25:58, 25.14s/it]Validation DataLoader 0:  34%|███▍      | 32/94 [14:31<25:58, 25.14s/it]Validation DataLoader 0:  35%|███▌      | 33/94 [14:44<21:47, 21.43s/it]Validation DataLoader 0:  35%|███▌      | 33/94 [14:44<21:47, 21.43s/it]Validation DataLoader 0:  36%|███▌      | 34/94 [15:53<35:50, 35.85s/it]Validation DataLoader 0:  36%|███▌      | 34/94 [15:53<35:50, 35.85s/it]Validation DataLoader 0:  37%|███▋      | 35/94 [16:04<27:57, 28.44s/it]Validation DataLoader 0:  37%|███▋      | 35/94 [16:04<27:57, 28.44s/it]Validation DataLoader 0:  38%|███▊      | 36/94 [16:15<22:27, 23.24s/it]Validation DataLoader 0:  38%|███▊      | 36/94 [16:15<22:27, 23.24s/it]Validation DataLoader 0:  39%|███▉      | 37/94 [16:43<23:21, 24.59s/it]Validation DataLoader 0:  39%|███▉      | 37/94 [16:43<23:21, 24.59s/it]Validation DataLoader 0:  40%|████      | 38/94 [17:15<25:04, 26.87s/it]Validation DataLoader 0:  40%|████      | 38/94 [17:15<25:04, 26.87s/it]Validation DataLoader 0:  41%|████▏     | 39/94 [17:51<27:08, 29.60s/it]Validation DataLoader 0:  41%|████▏     | 39/94 [17:51<27:08, 29.60s/it]Validation DataLoader 0:  43%|████▎     | 40/94 [18:12<24:19, 27.02s/it]Validation DataLoader 0:  43%|████▎     | 40/94 [18:12<24:19, 27.02s/it]Validation DataLoader 0:  44%|████▎     | 41/94 [18:32<21:56, 24.85s/it]Validation DataLoader 0:  44%|████▎     | 41/94 [18:32<21:56, 24.85s/it]Validation DataLoader 0:  45%|████▍     | 42/94 [18:42<17:41, 20.42s/it]Validation DataLoader 0:  45%|████▍     | 42/94 [18:42<17:41, 20.42s/it]Validation DataLoader 0:  46%|████▌     | 43/94 [19:09<18:56, 22.28s/it]Validation DataLoader 0:  46%|████▌     | 43/94 [19:09<18:56, 22.28s/it]Validation DataLoader 0:  47%|████▋     | 44/94 [21:23<46:30, 55.82s/it]Validation DataLoader 0:  47%|████▋     | 44/94 [21:23<46:30, 55.82s/it]Validation DataLoader 0:  48%|████▊     | 45/94 [22:35<49:31, 60.64s/it]Validation DataLoader 0:  48%|████▊     | 45/94 [22:35<49:31, 60.64s/it]Validation DataLoader 0:  49%|████▉     | 46/94 [24:13<57:38, 72.06s/it]Validation DataLoader 0:  49%|████▉     | 46/94 [24:13<57:38, 72.06s/it]Validation DataLoader 0:  50%|█████     | 47/94 [25:12<53:19, 68.07s/it]Validation DataLoader 0:  50%|█████     | 47/94 [25:12<53:19, 68.07s/it]Validation DataLoader 0:  51%|█████     | 48/94 [26:18<51:39, 67.37s/it]Validation DataLoader 0:  51%|█████     | 48/94 [26:18<51:39, 67.37s/it]Validation DataLoader 0:  52%|█████▏    | 49/94 [26:37<39:41, 52.92s/it]Validation DataLoader 0:  52%|█████▏    | 49/94 [26:37<39:41, 52.92s/it]Validation DataLoader 0:  53%|█████▎    | 50/94 [27:12<34:45, 47.39s/it]Validation DataLoader 0:  53%|█████▎    | 50/94 [27:12<34:45, 47.39s/it]Validation DataLoader 0:  54%|█████▍    | 51/94 [27:58<33:38, 46.95s/it]Validation DataLoader 0:  54%|█████▍    | 51/94 [27:58<33:38, 46.95s/it]Validation DataLoader 0:  55%|█████▌    | 52/94 [28:25<28:47, 41.12s/it]Validation DataLoader 0:  55%|█████▌    | 52/94 [28:25<28:47, 41.12s/it]Validation DataLoader 0:  56%|█████▋    | 53/94 [28:50<24:40, 36.12s/it]Validation DataLoader 0:  56%|█████▋    | 53/94 [28:50<24:40, 36.12s/it]Validation DataLoader 0:  57%|█████▋    | 54/94 [29:04<19:39, 29.48s/it]Validation DataLoader 0:  57%|█████▋    | 54/94 [29:04<19:39, 29.48s/it]Validation DataLoader 0:  59%|█████▊    | 55/94 [29:32<18:56, 29.14s/it]Validation DataLoader 0:  59%|█████▊    | 55/94 [29:32<18:56, 29.14s/it]Validation DataLoader 0:  60%|█████▉    | 56/94 [29:51<16:32, 26.13s/it]Validation DataLoader 0:  60%|█████▉    | 56/94 [29:51<16:32, 26.13s/it]Validation DataLoader 0:  61%|██████    | 57/94 [31:51<33:28, 54.30s/it]Validation DataLoader 0:  61%|██████    | 57/94 [31:51<33:28, 54.30s/it]Validation DataLoader 0:  62%|██████▏   | 58/94 [32:15<27:06, 45.18s/it]Validation DataLoader 0:  62%|██████▏   | 58/94 [32:15<27:06, 45.18s/it]Validation DataLoader 0:  63%|██████▎   | 59/94 [32:26<20:23, 34.96s/it]Validation DataLoader 0:  63%|██████▎   | 59/94 [32:26<20:23, 34.96s/it]Validation DataLoader 0:  64%|██████▍   | 60/94 [34:34<35:39, 62.92s/it]Validation DataLoader 0:  64%|██████▍   | 60/94 [34:34<35:39, 62.92s/it]Validation DataLoader 0:  65%|██████▍   | 61/94 [35:34<34:08, 62.09s/it]Validation DataLoader 0:  65%|██████▍   | 61/94 [35:34<34:08, 62.09s/it]Validation DataLoader 0:  66%|██████▌   | 62/94 [36:00<27:21, 51.29s/it]Validation DataLoader 0:  66%|██████▌   | 62/94 [36:00<27:21, 51.29s/it]Validation DataLoader 0:  67%|██████▋   | 63/94 [36:16<20:56, 40.52s/it]Validation DataLoader 0:  67%|██████▋   | 63/94 [36:16<20:56, 40.52s/it]Validation DataLoader 0:  68%|██████▊   | 64/94 [37:27<24:55, 49.86s/it]Validation DataLoader 0:  68%|██████▊   | 64/94 [37:27<24:55, 49.86s/it]Validation DataLoader 0:  69%|██████▉   | 65/94 [39:18<32:52, 68.03s/it]Validation DataLoader 0:  69%|██████▉   | 65/94 [39:18<32:52, 68.03s/it]Validation DataLoader 0:  70%|███████   | 66/94 [39:33<24:22, 52.22s/it]Validation DataLoader 0:  70%|███████   | 66/94 [39:33<24:22, 52.22s/it]Validation DataLoader 0:  71%|███████▏  | 67/94 [39:48<18:23, 40.88s/it]Validation DataLoader 0:  71%|███████▏  | 67/94 [39:48<18:23, 40.88s/it]Validation DataLoader 0:  72%|███████▏  | 68/94 [39:58<13:43, 31.67s/it]Validation DataLoader 0:  72%|███████▏  | 68/94 [39:58<13:43, 31.67s/it]Validation DataLoader 0:  73%|███████▎  | 69/94 [40:15<11:24, 27.37s/it]Validation DataLoader 0:  73%|███████▎  | 69/94 [40:15<11:24, 27.37s/it]Validation DataLoader 0:  74%|███████▍  | 70/94 [40:31<09:31, 23.83s/it]Validation DataLoader 0:  74%|███████▍  | 70/94 [40:31<09:31, 23.83s/it]Validation DataLoader 0:  76%|███████▌  | 71/94 [40:42<07:40, 20.01s/it]Validation DataLoader 0:  76%|███████▌  | 71/94 [40:42<07:40, 20.01s/it]Validation DataLoader 0:  77%|███████▋  | 72/94 [41:14<08:39, 23.60s/it]Validation DataLoader 0:  77%|███████▋  | 72/94 [41:14<08:39, 23.60s/it]Validation DataLoader 0:  78%|███████▊  | 73/94 [41:33<07:45, 22.17s/it]Validation DataLoader 0:  78%|███████▊  | 73/94 [41:33<07:45, 22.17s/it]Validation DataLoader 0:  79%|███████▊  | 74/94 [41:53<07:14, 21.74s/it]Validation DataLoader 0:  79%|███████▊  | 74/94 [41:53<07:14, 21.74s/it]Validation DataLoader 0:  80%|███████▉  | 75/94 [42:11<06:31, 20.61s/it]Validation DataLoader 0:  80%|███████▉  | 75/94 [42:11<06:31, 20.61s/it]Validation DataLoader 0:  81%|████████  | 76/94 [42:47<07:33, 25.17s/it]Validation DataLoader 0:  81%|████████  | 76/94 [42:47<07:33, 25.17s/it]Validation DataLoader 0:  82%|████████▏ | 77/94 [42:58<05:54, 20.85s/it]Validation DataLoader 0:  82%|████████▏ | 77/94 [42:58<05:54, 20.85s/it]Validation DataLoader 0:  83%|████████▎ | 78/94 [43:25<06:04, 22.78s/it]Validation DataLoader 0:  83%|████████▎ | 78/94 [43:25<06:04, 22.78s/it]Validation DataLoader 0:  84%|████████▍ | 79/94 [43:36<04:48, 19.25s/it]Validation DataLoader 0:  84%|████████▍ | 79/94 [43:36<04:48, 19.25s/it]Validation DataLoader 0:  85%|████████▌ | 80/94 [44:04<05:03, 21.69s/it]Validation DataLoader 0:  85%|████████▌ | 80/94 [44:04<05:03, 21.69s/it]Validation DataLoader 0:  86%|████████▌ | 81/94 [45:15<07:56, 36.68s/it]Validation DataLoader 0:  86%|████████▌ | 81/94 [45:15<07:56, 36.68s/it]Validation DataLoader 0:  87%|████████▋ | 82/94 [45:32<06:07, 30.66s/it]Validation DataLoader 0:  87%|████████▋ | 82/94 [45:32<06:07, 30.66s/it]Validation DataLoader 0:  88%|████████▊ | 83/94 [45:46<04:44, 25.84s/it]Validation DataLoader 0:  88%|████████▊ | 83/94 [45:46<04:44, 25.84s/it]Validation DataLoader 0:  89%|████████▉ | 84/94 [45:57<03:33, 21.34s/it]Validation DataLoader 0:  89%|████████▉ | 84/94 [45:57<03:33, 21.34s/it]Validation DataLoader 0:  90%|█████████ | 85/94 [46:08<02:43, 18.12s/it]Validation DataLoader 0:  90%|█████████ | 85/94 [46:08<02:43, 18.12s/it]Validation DataLoader 0:  91%|█████████▏| 86/94 [46:44<03:08, 23.62s/it]Validation DataLoader 0:  91%|█████████▏| 86/94 [46:44<03:08, 23.62s/it]Validation DataLoader 0:  93%|█████████▎| 87/94 [48:39<05:57, 51.02s/it]Validation DataLoader 0:  93%|█████████▎| 87/94 [48:39<05:57, 51.02s/it]Validation DataLoader 0:  94%|█████████▎| 88/94 [48:48<03:49, 38.22s/it]Validation DataLoader 0:  94%|█████████▎| 88/94 [48:48<03:49, 38.22s/it]Validation DataLoader 0:  95%|█████████▍| 89/94 [49:06<02:41, 32.38s/it]Validation DataLoader 0:  95%|█████████▍| 89/94 [49:06<02:41, 32.38s/it]Validation DataLoader 0:  96%|█████████▌| 90/94 [49:16<01:42, 25.66s/it]Validation DataLoader 0:  96%|█████████▌| 90/94 [49:16<01:42, 25.66s/it]Validation DataLoader 0:  97%|█████████▋| 91/94 [49:26<01:02, 20.74s/it]Validation DataLoader 0:  97%|█████████▋| 91/94 [49:26<01:02, 20.74s/it]Validation DataLoader 0:  98%|█████████▊| 92/94 [49:58<00:48, 24.32s/it]Validation DataLoader 0:  98%|█████████▊| 92/94 [49:58<00:48, 24.32s/it]Validation DataLoader 0:  99%|█████████▉| 93/94 [50:17<00:22, 22.51s/it]Validation DataLoader 0:  99%|█████████▉| 93/94 [50:17<00:22, 22.51s/it]Validation DataLoader 0: 100%|██████████| 94/94 [50:22<00:00, 17.41s/it]Validation DataLoader 0: 100%|██████████| 94/94 [50:22<00:00, 17.41s/it]Validation results saved to /n/fs/484-nlproofs/zs0608/NLProofS/prover/lightning_logs/version_18878425/results_val.json and /n/fs/484-nlproofs/zs0608/NLProofS/prover/lightning_logs/version_18878425/results_val.tsv

  0%|          | 0/187 [00:00<?, ?it/s][A100%|██████████| 187/187 [00:00<00:00, 2036.61it/s]
Performance by depth:
51 trees have depth 1
	leaves: 0.9411764705882353	0.9867413632119516
	steps: 0.8627450980392157	0.8627450980392157
	proof: 0.8627450980392157	0.8627450980392157
64 trees have depth 2
	leaves: 0.578125	0.9109093163780664
	steps: 0.40625	0.5097470238095239
	proof: 0.40625	0.40625
28 trees have depth 3
	leaves: 0.5	0.8999407338693052
	steps: 0.17857142857142858	0.3549799406942264
	proof: 0.17857142857142858	0.17857142857142858
18 trees have depth 4
	leaves: 0.5	0.8907106782106781
	steps: 0.1111111111111111	0.41101090267756935
	proof: 0.1111111111111111	0.1111111111111111
13 trees have depth 5
	leaves: 0.07692307692307693	0.7901144957289278
	steps: 0.0	0.2528155605078682
	proof: 0.0	0.0
7 trees have depth 6
	leaves: 0.0	0.6290464363883206
	steps: 0.0	0.15573152337858218
	proof: 0.0	0.0
6 trees have depth 7
	leaves: 0.16666666666666666	0.7525963738778384
	steps: 0.0	0.16511048863990038
	proof: 0.0	0.0
Performance by size:
41 trees have size 3
	leaves: 0.9512195121951219	0.9869918699186992
	steps: 0.9512195121951219	0.9512195121951219
	proof: 0.9512195121951219	0.9512195121951219
10 trees have size 4
	leaves: 0.9	0.9857142857142858
	steps: 0.5	0.5
	proof: 0.5	0.5
37 trees have size 5
	leaves: 0.5945945945945946	0.9053469053469053
	steps: 0.4864864864864865	0.563963963963964
	proof: 0.4864864864864865	0.4864864864864865
13 trees have size 6
	leaves: 0.5384615384615384	0.9150183150183151
	steps: 0.38461538461538464	0.5373626373626373
	proof: 0.38461538461538464	0.38461538461538464
26 trees have size 7
	leaves: 0.6153846153846154	0.9322344322344324
	steps: 0.3076923076923077	0.4567765567765568
	proof: 0.3076923076923077	0.3076923076923077
5 trees have size 8
	leaves: 0.6	0.9318181818181819
	steps: 0.0	0.14666666666666667
	proof: 0.0	0.0
10 trees have size 9
	leaves: 0.4	0.8968181818181818
	steps: 0.1	0.3015873015873016
	proof: 0.1	0.1
2 trees have size 10
	leaves: 0.0	0.7402597402597402
	steps: 0.0	0.43956043956043955
	proof: 0.0	0.0
14 trees have size 11
	leaves: 0.42857142857142855	0.8856524427953
	steps: 0.07142857142857142	0.33452380952380956
	proof: 0.07142857142857142	0.07142857142857142
1 trees have size 12
	leaves: 0.0	0.8571428571428571
	steps: 0.0	0.9090909090909091
	proof: 0.0	0.0
4 trees have size 13
	leaves: 0.75	0.9318181818181819
	steps: 0.0	0.303030303030303
	proof: 0.0	0.0
1 trees have size 15
	leaves: 0.0	0.4
	steps: 0.0	0.25
	proof: 0.0	0.0
2 trees have size 16
	leaves: 0.0	0.6941176470588235
	steps: 0.0	0.16025641025641024
	proof: 0.0	0.0
6 trees have size 17
	leaves: 0.0	0.6106893837156996
	steps: 0.0	0.07555189908131084
	proof: 0.0	0.0
2 trees have size 18
	leaves: 0.0	0.8750000000000001
	steps: 0.0	0.2532467532467533
	proof: 0.0	0.0
3 trees have size 19
	leaves: 0.0	0.6934001670843776
	steps: 0.0	0.2037037037037037
	proof: 0.0	0.0
1 trees have size 20
	leaves: 0.0	0.8571428571428571
	steps: 0.0	0.3333333333333333
	proof: 0.0	0.0
3 trees have size 21
	leaves: 0.0	0.7774337805297558
	steps: 0.0	0.11546840958605664
	proof: 0.0	0.0
3 trees have size 23
	leaves: 0.3333333333333333	0.9217273954116059
	steps: 0.0	0.33333333333333326
	proof: 0.0	0.0
2 trees have size 27
	leaves: 0.0	0.575657894736842
	steps: 0.0	0.12549019607843137
	proof: 0.0	0.0
1 trees have size 31
	leaves: 0.0	0.608695652173913
	steps: 0.0	0.09523809523809522
	proof: 0.0	0.0
Validation DataLoader 0: 100%|██████████| 94/94 [50:23<00:00, 32.16s/it]
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
     Validate metric           DataLoader 0
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  ExactMatch_leaves_val     0.5882352941176471
  ExactMatch_proof_val      0.4117647058823529
  ExactMatch_steps_val      0.4117647058823529
      F1_leaves_val         0.9039760977037425
      F1_proof_val          0.4117647058823529
      F1_steps_val          0.5311702138691758
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Configuration: 
 Namespace(config=None, subcommand='validate', validate=Namespace(config=[Path_fsr(cli_task2_stepwise_t5-large.yaml, cwd=/n/fs/484-nlproofs/zs0608/NLProofS/prover)], seed_everything=1, trainer=Namespace(logger=True, checkpoint_callback=None, enable_checkpointing=True, callbacks=[Namespace(class_path='pytorch_lightning.callbacks.LearningRateMonitor', init_args=Namespace(logging_interval='step', log_momentum=False))], default_root_dir=None, gradient_clip_val=0.5, gradient_clip_algorithm=None, process_position=0, num_nodes=1, num_processes=None, devices=None, gpus=1, auto_select_gpus=False, tpu_cores=None, ipus=None, log_gpu_memory=None, progress_bar_refresh_rate=None, enable_progress_bar=True, overfit_batches=0.0, track_grad_norm=-1, check_val_every_n_epoch=10, fast_dev_run=False, accumulate_grad_batches=32, max_epochs=600, min_epochs=None, max_steps=-1, min_steps=None, max_time=None, limit_train_batches=None, limit_val_batches=None, limit_test_batches=None, limit_predict_batches=None, val_check_interval=None, flush_logs_every_n_steps=None, log_every_n_steps=5, accelerator=None, strategy=None, sync_batchnorm=False, precision=32, enable_model_summary=True, weights_summary='top', weights_save_path=None, num_sanity_val_steps=2, resume_from_checkpoint=None, profiler=None, benchmark=None, deterministic=False, reload_dataloaders_every_n_epochs=0, auto_lr_find=False, replace_sampler_ddp=True, detect_anomaly=False, auto_scale_batch_size=False, prepare_data_per_node=None, plugins=None, amp_backend='native', amp_level=None, move_metrics_to_cpu=False, multiple_trainloader_mode='max_size_cycle', stochastic_weight_avg=False, terminate_on_nan=None), model=Namespace(stepwise=True, max_num_steps=20, model_name='t5-large', lr=5e-05, warmup_steps=1000, num_beams=10, topk=10, proof_search=True, verifier_weight=0.5, verifier_ckpt='/n/fs/484-nlproofs/weights/task2_verifier.ckpt', oracle_prover=False, oracle_verifier=False, dataset='entailmentbank', max_input_len=1024), data=Namespace(dataset='entailmentbank', sample_goal='intermediates', max_input_len=1024, max_output_len=64, batch_size=2, num_workers=2, path_train='../data/entailment_trees_emnlp2021_data_v3/dataset/task_2/train.jsonl', path_val='../data/entailment_trees_emnlp2021_data_v3/dataset/task_2/dev.jsonl', path_test='../data/entailment_trees_emnlp2021_data_v3/dataset/task_2/test.jsonl', subtree_proved_prob=0.75, subtree_proved_all_or_none=False, model_name='t5-large', stepwise=True), ckpt_path='/n/fs/484-nlproofs/weights/task2_prover.ckpt', verbose=True))
