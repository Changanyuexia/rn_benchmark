# ReleaseEval

## Dataset Overview

**ReleaseEval** is a comprehensive benchmark for evaluating the release note generation capabilities of Language Models, covering three core tasks:

- **commit2sum**: Generate release notes from commit messages.
- **tree2sum**: Generate release notes using commit tree structure information.
- **diff2sum**: Generate release notes based on code diffs between releases.

## Repository Structure

```
.
├── data_collection/   # Scripts for collecting release notes, commit messages, tags, and diffs from GitHub repositories. The repo.txt file specifies the target repositories.
├── clean/             # Scripts for cleaning and preprocessing release notes and commit data.
├── data/              # Datasets for model evaluation across the three tasks (commit2sum, tree2sum, diff2sum). Due to the large data limit, including CSV files for fine-tuning BART, T5, and few-shot LLMs. The _instruct.json files are specifically designed for fine-tuning LLMs.
├── experiments/       # Scripts and configs for model evaluation in both fine-tuning and few-shot settings.
├── human_eval/        # Human evaluation results and annotation files for dataset quality and model performance analysis, respectively.
├── results/           # Model evaluation results (few-shot and fine-tuning). Includes eval.py for automatic metric calculation (BLEU-4, ROUGE-L, METEOR).
└── README.md          # Project documentation.
```

## Models


|  Settings   | Models |
|--------------|--------|
| Fine-tuned   | BART, T5, [Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct), [Mistral-8B](https://huggingface.co/mistralai/Mistral-8B-Instruct-2410), [Llama3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) |
| Few-shot     | [Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct), [Mistral-8B](https://huggingface.co/mistralai/Mistral-8B-Instruct-2410), [Llama3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct), [Mistral-22B](https://huggingface.co/mistralai/Mistral-Small-Instruct-2409), [Qwen2.5-32B](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct), [Llama3.3-70B](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct), [Qwen2.5-72B](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct) |


## Getting Started

### Download the Data

Download the dataset from the `data/` directory.  

### Run Experiments

#### Fine-tuning

- We provide `job_bart.sh` and `job_t5.sh` for fine-tuning BART and T5 models on all three tasks.  
  Please download the corresponding pre-trained models into the `model/` directory before running.
- For fine-tuning LLMs such as Qwen2.5-7B, LLaMA3.1-8B, and Mistral-8B, we utilize the [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) framework.
- Example fine-tuning templates for Llama3.1-8B are provided in the experiments/ directory.
  
#### Few-shot

- Use `run_infer_all_tasks.sh` to perform few-shot inference with LLMs across all three tasks.
- The script will automatically evaluate models and save the results.


## Citation

If you use ReleaseEval in your research, please cite our work (citation coming soon).
