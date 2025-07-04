#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus-per-node=1
#SBATCH --time=06:05:00
#SBATCH --mem=60GB
#SBATCH --output=bart_result.log

source ~/.bashrc
conda activate LLM

python3 transformer-based/src/BART/run.py -t data/rn/train.csv -d data/rn/val.csv -e data/rn/test.csv -ms transformer-based/model/BART/rn -s result/BART/rn -epoch 3

