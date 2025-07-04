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

## Baseline Models

Models and scripts for the dataset tasks are located in the `baseline/` directory. The baseline models include:
- BART
- T5

LLM models include:
- Llama3.1
- mistral-v0.3
- Qwen

## Results

The `results/` directory contains all fine-tuning and few-shot results across three tasks.

- `eval.py`: A script for measuring model performance using metrics such as BLEU-4, ROUGE-L, and METEOR.

## Getting Started

### Download the Data

Download and unzip the dataset from the `data/` directory. We support the sample data here.

### Run experiments
we provide job_bart.sh and job_t5.sh for fine-tuning with bart and t5 across three tasks.
For fine-tuning with LLMs including Qwen2.5-7B, LLaMA3.1-8B, and Mistral-8B, we utilize the framework provided by https://github.com/hiyouga/LLaMA-Factory.
And run_infer_all_tasks.sh is for few-shot with LLMs across three tasks.
