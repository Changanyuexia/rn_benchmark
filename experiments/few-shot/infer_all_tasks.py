#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import json
import argparse
import logging
from datetime import datetime
import torch
from vllm import LLM, SamplingParams
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge
import nltk
from transformers import AutoTokenizer
import concurrent.futures
import csv

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Task configuration
tasks = {
    "tree2sum": {
        "data_dir": "../../data/tree2sum/",
        "prompt_instruction": (
            "Based on the given commit tree, generate a release note summarizing the key changes including bug fixes, new features, enhancements and other changes. "
            "The commit tree shows the sequence and relationship of commits using symbols: "
            "* = a commit, |\\ = branch split, |/ = branch merge, |* = commit on a side branch."
        ),
        "few_shot_template": """
Example {example_num}:
Commit Tree:
{commits}
Release Note:
{release}
""",
        "current_template": """
Commit Tree:
{commits}
Release Note:
""",
        "get_input": lambda ex: ex["commits"],
    },
    "commit2sum": {
        "data_dir": "../../data/commit2sum/",
        "prompt_instruction": (
            "Based on the given commit messages, generate a release note summarizing the key changes including bug fixes, new features, enhancements and other changes."
        ),
        "few_shot_template": """
Example {example_num}:
Commit Messages:
{commits}
Release Note:
{release}
""",
        "current_template": """
Commit Messages:
{commits}
Release Note:
""",
        "get_input": lambda ex: ex["commits"],
    },
    "diff2sum": {
        "data_dir": "../../data/diff2sum/",
        "prompt_instruction": (
            "Based on the given code diff including file modification and diff, generate a release note summarizing the key changes including bug fixes, new features, enhancements and other changes."
        ),
        "few_shot_template": """
Example {example_num}:
Mod & Diff:
{diffs}
Release Note:
{release}
""",
        "current_template": """
Mod & Diff:
{diffs}
Release Note:
""",
        "get_input": lambda ex: ex["diffs"],
    },
}

def read_csv_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data

def get_model_max_length(model_name):
    model_name = model_name.lower()
    if "qwen" in model_name or "mistral" in model_name:
        return 32768
    if "codellama" in model_name:
        return 16384
    if "llama" in model_name or "codegemma" in model_name:
        return 8192
    return 8192

def build_prompt(task_cfg, few_shot_examples, current_input, model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    max_tokens = get_model_max_length(model_name)
    safety_margin = 512
    prompt = task_cfg["prompt_instruction"] + "\n\n"
    used_tokens = len(tokenizer.encode(prompt))
    if few_shot_examples:
        intro = f"I will provide you with {len(few_shot_examples)} example{'s' if len(few_shot_examples) > 1 else ''}:\n\n"
        prompt += intro
        used_tokens += len(tokenizer.encode(intro))
        for i, ex in enumerate(few_shot_examples):
            example_block = task_cfg["few_shot_template"].format(
                example_num=i+1,
                commits=ex["commits"].strip(),
                release=ex["release"].strip()
            ) + "\n"
            block_tokens = len(tokenizer.encode(example_block))
            if used_tokens + block_tokens > max_tokens - safety_margin:
                break
            prompt += example_block
            used_tokens += block_tokens
    current_block = task_cfg["current_template"].format(
        commits=current_input.strip()
    )
    block_tokens = len(tokenizer.encode(current_block))
    if used_tokens + block_tokens > max_tokens - safety_margin:
        lines = current_input.strip().split('\n')
        truncated_lines = []
        for line in lines:
            trial_input = '\n'.join(truncated_lines + [line])
            trial_block = task_cfg["current_template"].format(
                commits=trial_input
            )
            if used_tokens + len(tokenizer.encode(trial_block)) > max_tokens - safety_margin:
                break
            truncated_lines.append(line)
        truncated_text = '\n'.join(truncated_lines) + '\n... (truncated)'
        current_block = task_cfg["current_template"].format(
            commits=truncated_text
        )
    prompt += current_block
    return prompt

def compute_metrics(preds, refs):
    score_dict = {"rouge-l": [], "bleu-4": [], "meteor": []}
    rouge = Rouge()
    smoothing = SmoothingFunction().method3
    def single_eval(args):
        pred, ref = args
        if not pred.strip():
            return 0.0, 0.0, 0.0
        hypothesis = nltk.word_tokenize(pred.lower())
        reference = nltk.word_tokenize(ref.lower())
        scores = rouge.get_scores(" ".join(hypothesis), " ".join(reference))
        rouge_l = round(scores[0]["rouge-l"]["f"] * 100, 4)
        try:
            bleu_score = nltk.translate.bleu_score.corpus_bleu(
                [[reference]], [hypothesis],
                smoothing_function=smoothing,
                weights=(0.25, 0.25, 0.25, 0.25)
            )
            bleu = round(bleu_score * 100, 4)
        except Exception:
            bleu = 0.0
        try:
            meteor = meteor_score([reference], hypothesis)
            meteor = round(meteor * 100, 4)
        except Exception:
            meteor = 0.0
        return rouge_l, bleu, meteor
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(single_eval, zip(preds, refs)))
    for rouge_l, bleu, meteor in results:
        score_dict["rouge-l"].append(rouge_l)
        score_dict["bleu-4"].append(bleu)
        score_dict["meteor"].append(meteor)
    return {k: round(sum(v) / len(v), 4) for k, v in score_dict.items()}

def run_task(task_name, args, llm):
    task_cfg = tasks[task_name]
    # Support train/val/test
    split = args.split
    data_file = os.path.join(task_cfg["data_dir"], f"{split}_sample.csv")
    test_data = read_csv_data(data_file)
    prompt_data = test_data[:args.num_few_shot] if args.num_few_shot > 0 else []
    few_shot_examples = prompt_data
    if args.test_size is not None:
        test_data = test_data[:args.test_size]
        logger.info(f"Processing first {args.test_size} test examples for {task_name}")
    logger.info(f"Using {len(few_shot_examples)} few-shot examples for {task_name}")
    predictions = []
    total_examples = len(test_data)
    logger.info(f"Starting inference on {total_examples} examples for {task_name}")
    model_name = args.model
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    max_tokens = get_model_max_length(model_name)
    logger.info(f"[Task: {task_name}] Using tokenizer: {model_name}, max_tokens: {max_tokens}")
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop=args.stop if args.stop else None
    )
    prompts = []
    for idx, example in enumerate(test_data):
        current_input = task_cfg["get_input"](example)
        prompt = build_prompt(task_cfg, few_shot_examples, current_input, model_name)
        if prompt is None:
            logger.error(f"[Task: {task_name}] Skipping example {idx} due to length limit")
            continue
        logger.info(f"[Task: {task_name}] Example {idx} prompt length: {len(prompt)} | Prompt preview: {prompt[:200].replace(chr(10),' ')} ...")
        prompts.append(prompt)
    batch_size = 100
    num_batches = (len(prompts) + batch_size - 1) // batch_size
    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min((batch_idx + 1) * batch_size, len(prompts))
        batch_prompts = prompts[start:end]
        logger.info(f"[Task: {task_name}] Running batch {batch_idx+1}/{num_batches} ({start}-{end-1})")
        try:
            outputs = llm.generate(batch_prompts, sampling_params)
        except Exception as e:
            logger.error(f"[Task: {task_name}] vLLM generation failed in batch {batch_idx+1}: {str(e)}")
            continue
        sorted_outputs = sorted(outputs, key=lambda o: int(o.request_id))
        for i, (example, out) in enumerate(zip(test_data[start:end], sorted_outputs)):
            prediction = out.outputs[0].text.strip()
            if "Release Note:" in prediction:
                prediction = prediction.split("Release Note:")[-1].strip()
            logger.info(f"[Task: {task_name}] Example {start+i} prediction preview: {prediction[:200].replace(chr(10),' ')} ...")
            predictions.append({
                "release": example["release"],
                "prediction": prediction
            })
    os.makedirs(os.path.join(task_cfg["data_dir"], "output"), exist_ok=True)
    output_file = os.path.join(
        task_cfg["data_dir"], "output",
        f"{args.model.split('/')[-1]}_{split}_few_shot_{args.num_few_shot}.json"
    )
    with open(output_file, "w") as f:
        json.dump(predictions, f, indent=2)
    logger.info(f"[Task: {task_name}] Saved predictions to {output_file}")
    preds = [p["prediction"] for p in predictions]
    refs = [p["release"] for p in predictions]
    metrics = compute_metrics(preds, refs)
    logger.info(f"[Task: {task_name}] Evaluation Metrics:")
    logger.info(f"[Task: {task_name}] BLEU-4: {metrics['bleu-4']:.4f}")
    logger.info(f"[Task: {task_name}] ROUGE-L: {metrics['rouge-l']:.4f}")
    logger.info(f"[Task: {task_name}] METEOR: {metrics['meteor']:.4f}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(task_cfg["data_dir"], "output", "evaluation_results.csv")
    header = ["timestamp", "model", "num_few_shot", "split", "bleu-4", "rouge-l", "meteor"]
    row = {
        "timestamp": timestamp,
        "model": args.model.split("/")[-1],
        "num_few_shot": args.num_few_shot,
        "split": split,
        "bleu-4": metrics["bleu-4"],
        "rouge-l": metrics["rouge-l"],
        "meteor": metrics["meteor"]
    }
    write_header = not os.path.exists(csv_path)
    with open(csv_path, mode='a', newline='', encoding='utf-8') as f:
        import csv as csvlib
        writer = csvlib.DictWriter(f, fieldnames=header)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    logger.info(f"[Task: {task_name}] Saved evaluation results to {csv_path}")
    logger.info(f"========== [END] Task: {task_name} | Model: {args.model} | Few-shot: {args.num_few_shot} | Split: {split} ==========")

def parse_args():
    parser = argparse.ArgumentParser(description="Unified Release Note Generation for All Tasks")
    parser.add_argument("--model", type=str, required=True, help="Model path or HuggingFace ID")
    parser.add_argument("--num_few_shot", type=int, choices=[0,1,2,4], default=0, help="Number of few-shot examples")
    parser.add_argument("--max_batched_tokens", type=int, default=4096, help="vLLM max_num_batched_tokens and max_model_len")
    parser.add_argument("--max_tokens", type=int, default=256, help="Max tokens to generate per example")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p sampling probability")
    parser.add_argument("--stop", nargs="+", default=None, help="Stop sequences for generation (optional)")
    parser.add_argument("--test_size", type=int, default=None, help="Number of test examples to process (default: all)")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Which split to use")
    parser.add_argument("--only_task", type=str, default=None, choices=["tree2sum", "commit2sum", "diff2sum"], help="Only run the specified task")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    num_gpus = torch.cuda.device_count()
    model_max_len = get_model_max_length(args.model)
    logger.info(f"[Global] Using model: {args.model}, max_model_len: {model_max_len}")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=max(1, num_gpus),
        max_num_batched_tokens=model_max_len,
        max_model_len=model_max_len,
        gpu_memory_utilization=0.9,
        swap_space=4,
        block_size=16,
        trust_remote_code=True,
        tokenizer_mode="auto",
        enforce_eager=True,
    )
    tasks_to_run = [args.only_task] if args.only_task else ["tree2sum", "commit2sum", "diff2sum"]
    for task_name in tasks_to_run:
        run_task(task_name, args, llm)

if __name__ == "__main__":
    main() 
