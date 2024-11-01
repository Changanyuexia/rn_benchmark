# CommitsSum Dataset

## Dataset Overview

The CommitsSum benchmark dataset supports three distinct release note generation tasks:

- **commits2sum**: Generates release notes from commit messages.
- **tree2sum**: Generates release notes using commit tree information.
- **diff2sum**: Generates release notes based on code differences.

## Baseline Models

Models and scripts for the dataset tasks are located in the `baseline/` directory. The baseline models include:
- BART
- T5

LLM models include:
- Llama3.1
- mistral-v0.3
- Qwen

## Results

The `results/` directory contains the human evaluation results.

- `eval.py`: A script for measuring model performance using metrics such as BLEU-4, ROUGE-L, and METEOR.

## Getting Started

### Download the Dataset

Download and unzip the dataset from the `dataset/` directory. We support the sample data here.

### Run Baselines

Use the provided scripts in `baseline/` to reproduce the baseline results. Below is an example instruction to run baselines. Note that you need to download the model BART from the official site.

```bash
python3 baselines/BART/run.py -t dataset/diff_rn/train.csv -d dataset/diff_rn/val.csv -e dataset/diff_rn/test.csv -ms model/BART/diff_rn -s result/BART/diff_rn -epoch 5

