/n/fs/484-nlproofs/miniconda3/envs/nlproofs2/lib/python3.9/site-packages/ete3-3.1.2-py3.7.egg/ete3/evol/parser/codemlparser.py:221: SyntaxWarning: "is" with a literal. Did you mean "=="?
/n/fs/484-nlproofs/miniconda3/envs/nlproofs2/lib/python3.9/site-packages/ete3-3.1.2-py3.7.egg/ete3/evol/parser/codemlparser.py:221: SyntaxWarning: "is" with a literal. Did you mean "=="?
Global seed set to 1
Some weights of the model checkpoint at roberta-large were not used when initializing RobertaModel: ['lm_head.dense.bias', 'lm_head.layer_norm.bias', 'lm_head.dense.weight', 'lm_head.layer_norm.weight', 'lm_head.bias', 'lm_head.decoder.weight']
- This IS expected if you are initializing RobertaModel from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing RobertaModel from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
/n/fs/484-nlproofs/miniconda3/envs/nlproofs2/lib/python3.9/site-packages/torchmetrics/utilities/prints.py:36: UserWarning: Metric `AveragePrecision` will save all targets and predictions in buffer. For large datasets this may lead to large memory footprint.
  warnings.warn(*args, **kwargs)
GPU available: True, used: True
TPU available: False, using: 0 TPU cores
IPU available: False, using: 0 IPUs
HPU available: False, using: 0 HPUs
Restoring states from the checkpoint path at /n/fs/484-nlproofs/weights/task2_prover.ckpt
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Loaded model weights from checkpoint at /n/fs/484-nlproofs/weights/task2_prover.ckpt
/n/fs/484-nlproofs/miniconda3/envs/nlproofs2/lib/python3.9/site-packages/pytorch_lightning/trainer/connectors/data_connector.py:487: PossibleUserWarning: Your `val_dataloader`'s sampler has shuffling enabled, it is strongly recommended that you turn shuffling off for val/test/predict dataloaders.
  rank_zero_warn(
/n/fs/484-nlproofs/miniconda3/envs/nlproofs2/lib/python3.9/site-packages/pytorch_lightning/trainer/connectors/data_connector.py:240: PossibleUserWarning: The dataloader, val_dataloader 0, does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` (try 104 which is the number of cpus on this machine) in the `DataLoader` init to improve performance.
  rank_zero_warn(
187 proofs loaded. 0 invalid ones removed.
Validation: 0it [00:00, ?it/s]Validation:   0%|          | 0/10 [00:00<?, ?it/s]Validation DataLoader 0:   0%|          | 0/10 [00:00<?, ?it/s]Validation DataLoader 0:  10%|█         | 1/10 [00:05<00:47,  5.31s/it]Validation DataLoader 0:  10%|█         | 1/10 [00:05<00:47,  5.31s/it]Validation DataLoader 0:  20%|██        | 2/10 [00:07<00:25,  3.18s/it]Validation DataLoader 0:  20%|██        | 2/10 [00:07<00:25,  3.18s/it]Validation DataLoader 0:  30%|███       | 3/10 [00:10<00:22,  3.27s/it]Validation DataLoader 0:  30%|███       | 3/10 [00:10<00:22,  3.27s/it]