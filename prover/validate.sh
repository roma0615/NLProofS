#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --mail-user=zs0608@cs.princeton.edu
#SBATCH --mail-type=ALL
#SBATCH --output=slurm/%j.sh

source /n/fs/484-nlproofs/miniconda3/bin/activate
source activate nlproofs2

# Train a single-shot prover on Task 2 of EntailmentBank.
# python main.py fit --config cli_task2_stepwise_t5-small.yaml 

# Validate NLProofS (stepwise prover + verifier-guided search).
# python main.py validate --config cli_task2_stepwise_t5-large.yaml --ckpt_path /n/fs/484-nlproofs/weights/task2_prover.ckpt --model.verifier_weight 0.5 --model.verifier_ckpt /n/fs/484-nlproofs/weights/task2_verifier.ckpt --model.proof_search true

# Validate NLProofs (oracle prover + verifier-guided search).
python main.py validate --config cli_task2_stepwise_t5-large.yaml --ckpt_path /n/fs/484-nlproofs/weights/task2_prover.ckpt --model.verifier_weight 0.5 --model.verifier_ckpt /n/fs/484-nlproofs/weights/task2_verifier.ckpt --model.proof_search true --model.gpt_prover true --model.topk 5 --trainer.limit_val_batches 10

# Validate NLProofS (gpt prover + verifier-guided search).
python main.py validate --config cli_task2_stepwise_t5-large.yaml --ckpt_path /n/fs/484-nlproofs/weights/task2_prover.ckpt --model.verifier_weight 0.5 --model.verifier_ckpt /n/fs/484-nlproofs/weights/task2_verifier.ckpt --model.proof_search true --model.gpt_prover false --model.topk 5 --trainer.limit_val_batches 10

# Test NLProofS (gpt prover + verifier-guided search).
# python main.py test --config cli_task2_stepwise_t5-large.yaml --ckpt_path /n/fs/484-nlproofs/weights/task2_prover.ckpt --model.verifier_weight 0.5 --model.verifier_ckpt /n/fs/484-nlproofs/weights/task2_verifier.ckpt --model.proof_search true --model.gpt_prover true --model.topk 5 --trainer.limit_test_batches 10

# Test NLProofS (stepwise prover + verifier-guided search).
# python main.py test --config cli_task2_stepwise_t5-large.yaml --ckpt_path /n/fs/484-nlproofs/weights/task2_prover.ckpt --model.verifier_weight 0.5 --model.verifier_ckpt /n/fs/484-nlproofs/weights/task2_verifier.ckpt --model.proof_search true --model.gpt_prover false --model.topk 5 --trainer.limit_test_batches 10
