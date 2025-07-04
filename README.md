# CommitsSum Dataset

## Dataset Overview

The CommitsSum benchmark dataset supports three distinct release note generation tasks:

- **commits2sum**: Generates release notes from commit messages.
- **tree2sum**: Generates release notes using commit tree information.
- **diff2sum**: Generates release notes based on code differences.

## Directory Structure
requirement.txt includes the environment packages.

### data_collection

Contains scripts and utilities for fetching tags, release notes, and commit data from repositories. 

### clean

Includes scripts for cleaning and preprocessing the release notes and commit data. 

### data

Includes each dataset for model evaluation across three tasks

## Models

Fine-tuned models including
- BART
- T5
- Qwen2.5-7B https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
- Ministral-8B https://huggingface.co/mistralai/Ministral-8B-Instruct-2410
- Llama3.1-8B https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
  
Few-shot models including
- Qwen2.5-7B https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
- Ministral-8B https://huggingface.co/mistralai/Ministral-8B-Instruct-2410
- Llama3.1-8B https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
- Mistral-22B https://huggingface.co/mistralai/Mistral-Small-Instruct-2409
- Qwen2.5-32B https://huggingface.co/Qwen/Qwen2.5-32B-Instruct
- llama3.3-70B https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct
- Qwen2.5-72B https://huggingface.co/Qwen/Qwen2.5-72B-Instruct


## Results

The `results/` directory contains all fine-tuning and few-shot results across three tasks.

- `eval.py`: A script for measuring model performance using metrics such as BLEU-4, ROUGE-L, and METEOR.

## Getting Started

### Download the Data

Download and unzip the dataset from the `data/` directory. We support the sample data here.

### Run experiments
We provide job_bart.sh and job_t5.sh for fine-tuning with bart and t5 across three tasks.

For fine-tuning with LLMs including Qwen2.5-7B, LLaMA3.1-8B, and Mistral-8B, we utilize the framework provided by https://github.com/hiyouga/LLaMA-Factory.

And run_infer_all_tasks.sh is for few-shot with LLMs across three tasks.
