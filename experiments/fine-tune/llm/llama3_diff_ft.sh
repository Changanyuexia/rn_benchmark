#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:4
#SBATCH --time=15:05:00
#SBATCH --mem=720GB
#SBATCH --output=llama3_diff.log

source ~/.bashrc
conda activate lf


CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train examples/train_lora/llama3_diff_train.yaml
CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train examples/train_lora/llama3_diff_predict.yaml 
