# rn_benchmark_dataset

Dataset Overview
The rn_benchmark_dataset supports three distinct release note generation tasks:

commits2sum: Generates release notes from commit messages.
tree2sum: Generates release notes using commit tree information.
diff2sum: Generates release notes based on code differences.

Baseline Models

The baseline/ provides models and scripts to serve for the tasks within this dataset. 
Baseline models include: BART, T5
LLM models include: Llama3.1, mistral-v0.3, Qwen. We refer to the framework from https://github.com/hiyouga/LLaMA-Factory to run LLMs.


Results

The results/ directory contains the human evaluation results.
eval.py: script for measuring model performance using metrics such as BLEU-4, ROUGE-L, and METEOR.


Getting Started

Download the Dataset: Download and unzip the dataset from dataset/.
Run Baselines: Use the provided scripts in baseline/ to reproduce the baseline results.
Run LLMs: using https://github.com/hiyouga/LLaMA-Factory to run Llama3.1, mistral-v0.3, Qwen.
