#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus-per-node=1
#SBATCH --time=06:00:00
#SBATCH --mem=60GB
#SBATCH --output=t5_result.log

source ~/.bashrc
conda activate LLM

python3 transformer-based/src/T5/run.py -t data/rn/train.csv -d data/rn/val.csv -e data/rn/test.csv -ms transformer-based/model/T5/rn -s result/T5/rn -epoch 3

