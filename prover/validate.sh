#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --mail-user=zs0608@cs.princeton.edu
#SBATCH --mail-type=ALL
#SBATCH --output=slurm/%j.sh

source /n/fs/484-nlproofs/miniconda3/bin/activate
source activate nlproofs

# Validate the stepwise prover without verifier-guided search on Task 2 of EntailmentBank.
# python main.py validate --config cli_task2_stepwise_t5-large.yaml --ckpt_path /n/fs/484-nlproofs/weights/task2_prover.ckpt

# Validate NLProofS (stepwise prover + verifier-guided search).
python main.py validate --config cli_task2_stepwise_t5-large.yaml --ckpt_path /n/fs/484-nlproofs/weights/task2_prover.ckpt --model.verifier_weight 0.5 --model.verifier_ckpt /n/fs/484-nlproofs/weights/task2_verifier.ckpt --model.proof_search true

# Validate NLProofS w/o prover score.
# python main.py validate --config cli_task2_stepwise_t5-large.yaml --ckpt_path /n/fs/484-nlproofs/weights/task2_prover.ckpt --model.verifier_weight 1.0 --model.verifier_ckpt /n/fs/484-nlproofs/weights/task2_verifier.ckpt --model.proof_search true