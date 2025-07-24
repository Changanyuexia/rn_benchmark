# ReleaseEval Dataset

## Dataset Overview

**ReleaseEval** is a comprehensive benchmark for evaluating the release note generation capabilities of Language Models (LMs), covering three core tasks:

- **commit2sum**: Generate release notes from commit messages.
- **tree2sum**: Generate release notes using commit tree structure information.
- **diff2sum**: Generate release notes based on code diffs between releases.

## Repository Structure

```
.
├── data_collection/   # Scripts for collecting release notes, commit messages, tags, and diffs from GitHub repositories. The repo.txt file specifies the target repositories.
├── clean/             # Scripts for cleaning and preprocessing release notes and commit data.
├── data/              # Datasets for model evaluation across the three tasks (commit2sum, tree2sum, diff2sum). Includes both full and sample data.
├── experiments/       # Scripts and configs for model evaluation in both fine-tuning and few-shot settings.
├── human_eval/        # Human evaluation results and annotation files for dataset quality and model performance analysis.
├── results/           # Model evaluation results (few-shot and fine-tuning). Includes eval.py for automatic metric calculation (BLEU-4, ROUGE-L, METEOR).
└── README.md          # Project documentation.
```

## Models

### Fine-tuned Models

- BART
- T5
- Qwen2.5-7B ([HuggingFace](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct))
- Mistral-8B ([HuggingFace](https://huggingface.co/mistralai/Mistral-8B-Instruct-2410))
- Llama3.1-8B ([HuggingFace](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct))

### Few-shot Models

- Qwen2.5-7B ([HuggingFace](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct))
- Mistral-8B ([HuggingFace](https://huggingface.co/mistralai/Mistral-8B-Instruct-2410))
- Llama3.1-8B ([HuggingFace](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct))
- Mistral-22B ([HuggingFace](https://huggingface.co/mistralai/Mistral-Small-Instruct-2409))
- Qwen2.5-32B ([HuggingFace](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct))
- Llama3.3-70B ([HuggingFace](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct))
- Qwen2.5-72B ([HuggingFace](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct))

## Getting Started

### Download the Data

Download and unzip the dataset from the `data/` directory.  
Due to the large size of the full dataset, we also provide sample data for quick testing and development.

### Data Collection & Preprocessing

- Use scripts in `data_collection/` to crawl and process release notes, commit messages, tags, and diffs from GitHub repositories listed in `repo.txt`.
- Use scripts in `clean/` to clean and filter the raw data, producing high-quality release note datasets for each task.

### Run Experiments

#### Fine-tuning

- We provide `job_bart.sh` and `job_t5.sh` for fine-tuning BART and T5 models on all three tasks.  
  Please download the corresponding pre-trained models into the `model/` directory before running.
- For fine-tuning LLMs such as Qwen2.5-7B, LLaMA3.1-8B, and Mistral-8B, we utilize the [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) framework.
- Example fine-tuning templates for Llama3.1-8B and other LLMs are provided in the `experiments/` directory.

#### Few-shot Inference

- Use `run_infer_all_tasks.sh` to perform few-shot inference with LLMs across all three tasks.
- The script will automatically evaluate models using the sample or full datasets and save the results.

### Human Evaluation

- Human evaluation results and annotation files are provided in the `human_eval/` directory for both dataset quality and model performance analysis.

### Automatic Evaluation

- Use `results/eval.py` to compute automatic metrics (BLEU-4, ROUGE-L, METEOR) for model predictions.

## Citation

If you use ReleaseEval in your research, please cite our work (citation coming soon).
