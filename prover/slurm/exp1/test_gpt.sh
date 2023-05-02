whoami: cannot find name for user ID 227186
/n/fs/484-nlproofs/miniconda3/envs/nlproofs2/lib/python3.9/site-packages/ete3-3.1.2-py3.7.egg/ete3/evol/parser/codemlparser.py:221: SyntaxWarning: "is" with a literal. Did you mean "=="?
/n/fs/484-nlproofs/miniconda3/envs/nlproofs2/lib/python3.9/site-packages/ete3-3.1.2-py3.7.egg/ete3/evol/parser/codemlparser.py:221: SyntaxWarning: "is" with a literal. Did you mean "=="?
Global seed set to 1
Some weights of the model checkpoint at roberta-large were not used when initializing RobertaModel: ['lm_head.layer_norm.bias', 'lm_head.layer_norm.weight', 'lm_head.dense.weight', 'lm_head.decoder.weight', 'lm_head.dense.bias', 'lm_head.bias']
- This IS expected if you are initializing RobertaModel from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing RobertaModel from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
/n/fs/484-nlproofs/miniconda3/envs/nlproofs2/lib/python3.9/site-packages/torchmetrics/utilities/prints.py:36: UserWarning: Metric `AveragePrecision` will save all targets and predictions in buffer. For large datasets this may lead to large memory footprint.
  warnings.warn(*args, **kwargs)
Multiprocessing is handled by SLURM.
GPU available: True, used: True
TPU available: False, using: 0 TPU cores
IPU available: False, using: 0 IPUs
HPU available: False, using: 0 HPUs
Restoring states from the checkpoint path at /n/fs/484-nlproofs/weights/task2_prover.ckpt
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Loaded model weights from checkpoint at /n/fs/484-nlproofs/weights/task2_prover.ckpt
/n/fs/484-nlproofs/miniconda3/envs/nlproofs2/lib/python3.9/site-packages/pytorch_lightning/trainer/connectors/data_connector.py:487: PossibleUserWarning: Your `test_dataloader`'s sampler has shuffling enabled, it is strongly recommended that you turn shuffling off for val/test/predict dataloaders.
  rank_zero_warn(
/n/fs/484-nlproofs/miniconda3/envs/nlproofs2/lib/python3.9/site-packages/pytorch_lightning/trainer/connectors/data_connector.py:240: PossibleUserWarning: The dataloader, test_dataloader 0, does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` (try 104 which is the number of cpus on this machine) in the `DataLoader` init to improve performance.
  rank_zero_warn(
340 proofs loaded. 0 invalid ones removed.

To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)



Performance by depth:
4 trees have depth 1
	leaves: 1.0	1.0
	steps: 0.75	0.9166666666666666
	proof: 0.75	0.75
6 trees have depth 2
	leaves: 0.5	0.7952380952380954
	steps: 0.3333333333333333	0.4166666666666667
	proof: 0.3333333333333333	0.3333333333333333
4 trees have depth 3
	leaves: 0.0	0.452020202020202
	steps: 0.0	0.0
	proof: 0.0	0.0
1 trees have depth 4
	leaves: 0.0	0.6666666666666666
	steps: 0.0	0.28571428571428575
	proof: 0.0	0.0
5 trees have depth 5
	leaves: 0.4	0.7607843137254902
	steps: 0.0	0.17714285714285713
	proof: 0.0	0.0
Performance by size:
4 trees have size 3
	leaves: 1.0	1.0
	steps: 0.75	0.9166666666666666
	proof: 0.75	0.75
4 trees have size 5
	leaves: 0.5	0.7428571428571429
	steps: 0.5	0.5
	proof: 0.5	0.5
1 trees have size 6
	leaves: 1.0	1.0
	steps: 0.0	0.5
	proof: 0.0	0.0
3 trees have size 7
	leaves: 0.0	0.49898989898989904
	steps: 0.0	0.0
	proof: 0.0	0.0
2 trees have size 9
	leaves: 0.0	0.5555555555555555
	steps: 0.0	0.0
	proof: 0.0	0.0
2 trees have size 11
	leaves: 0.5	0.8333333333333333
	steps: 0.0	0.3
	proof: 0.0	0.0
1 trees have size 14
	leaves: 0.0	0.6666666666666666
	steps: 0.0	0.28571428571428575
	proof: 0.0	0.0
1 trees have size 15
	leaves: 1.0	1.0
	steps: 0.0	0.0
	proof: 0.0	0.0
1 trees have size 17
	leaves: 0.0	0.6666666666666666
	steps: 0.0	0.0
	proof: 0.0	0.0
1 trees have size 23
	leaves: 0.0	0.47058823529411764
	steps: 0.0	0.28571428571428575
	proof: 0.0	0.0

────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
       Test metric             DataLoader 0
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
 ExactMatch_leaves_test            0.45
  ExactMatch_proof_test            0.25
  ExactMatch_steps_test            0.25
     F1_leaves_test         0.7525048807401749
      F1_proof_test                0.25
      F1_steps_test         0.36690476190476196
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Configuration: 
 Namespace(config=None, subcommand='test', test=Namespace(config=[Path_fsr(cli_task2_stepwise_t5-large.yaml, cwd=/n/fs/484-nlproofs/zs0608/NLProofS/prover)], seed_everything=1, trainer=Namespace(logger=True, checkpoint_callback=None, enable_checkpointing=True, callbacks=[Namespace(class_path='pytorch_lightning.callbacks.LearningRateMonitor', init_args=Namespace(logging_interval='step', log_momentum=False))], default_root_dir=None, gradient_clip_val=0.5, gradient_clip_algorithm=None, process_position=0, num_nodes=1, num_processes=None, devices=None, gpus=1, auto_select_gpus=False, tpu_cores=None, ipus=None, log_gpu_memory=None, progress_bar_refresh_rate=None, enable_progress_bar=True, overfit_batches=0.0, track_grad_norm=-1, check_val_every_n_epoch=10, fast_dev_run=False, accumulate_grad_batches=32, max_epochs=600, min_epochs=None, max_steps=-1, min_steps=None, max_time=None, limit_train_batches=None, limit_val_batches=None, limit_test_batches=10, limit_predict_batches=None, val_check_interval=None, flush_logs_every_n_steps=None, log_every_n_steps=5, accelerator=None, strategy=None, sync_batchnorm=False, precision=32, enable_model_summary=True, weights_summary='top', weights_save_path=None, num_sanity_val_steps=2, resume_from_checkpoint=None, profiler=None, benchmark=None, deterministic=False, reload_dataloaders_every_n_epochs=0, auto_lr_find=False, replace_sampler_ddp=True, detect_anomaly=False, auto_scale_batch_size=False, prepare_data_per_node=None, plugins=None, amp_backend='native', amp_level=None, move_metrics_to_cpu=False, multiple_trainloader_mode='max_size_cycle', stochastic_weight_avg=False, terminate_on_nan=None), model=Namespace(stepwise=True, max_num_steps=20, model_name='t5-large', lr=5e-05, warmup_steps=1000, num_beams=10, topk=5, proof_search=True, verifier_weight=0.5, verifier_ckpt='/n/fs/484-nlproofs/weights/task2_verifier.ckpt', oracle_prover=False, oracle_verifier=False, gpt_prover=True, gpt_context_size=8, gpt_examples_file='examples.txt', dataset='entailmentbank', max_input_len=1024), data=Namespace(dataset='entailmentbank', sample_goal='intermediates', max_input_len=1024, max_output_len=64, batch_size=2, num_workers=2, path_train='../data/entailment_trees_emnlp2021_data_v3/dataset/task_2/train.jsonl', path_val='../data/entailment_trees_emnlp2021_data_v3/dataset/task_2/dev.jsonl', path_test='../data/entailment_trees_emnlp2021_data_v3/dataset/task_2/test.jsonl', subtree_proved_prob=0.75, subtree_proved_all_or_none=False, model_name='t5-large', stepwise=True), ckpt_path='/n/fs/484-nlproofs/weights/task2_prover.ckpt', verbose=True))