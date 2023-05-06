#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --mail-user=rb4785@princeton.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name=stepwise
#SBATCH --output=%x_%j.log

source /n/fs/484-nlproofs/miniconda3/bin/activate
conda activate nlproofs2
cd ../prover

# run stepwise model trained on task2, on task3
# python main.py test --config cli_task3_stepwise_t5-large.yaml --ckpt_path ../../../weights/task2_prover.ckpt --model.verifier_ckpt ../../../weights/task2_verifier.ckpt --model.proof_search true
python main.py validate --config cli_task3_stepwise_t5-large.yaml --ckpt_path ../../../weights/task2_prover.ckpt --model.verifier_ckpt ../../../weights/task2_verifier.ckpt --model.proof_search true

