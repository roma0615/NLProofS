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