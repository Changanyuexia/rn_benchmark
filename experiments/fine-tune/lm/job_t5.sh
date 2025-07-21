#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus-per-node=1
#SBATCH --time=06:05:00
#SBATCH --mem=60GB
#SBATCH --output=bart_result.log

source ~/.bashrc
conda activate LLM

python3 src/T5/run.py -t rn_benchmark/data/commit2sum/train.csv -d rn_benchmark/data/commit2sum/val.csv -e rn_benchmark/data/commit2sum/test.csv -ms model/T5/rn -s rn_benchmark/result/T5/fine-tune/commit2sum -epoch 3

