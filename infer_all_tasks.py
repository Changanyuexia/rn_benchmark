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

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# 任务配置
TASKS = {
    "graph_rn": {
        "test_file": "../graph_rn/test.json",
        "prompt_file": "../graph_rn/prompt.json",
        "output_dir": "output/graph_rn/",
        "prompt_instruction": (
            "Based on the given commit tree, generate a release note summarizing the key changes including bug fixes, new features, enhancements and other changes. "
            "The commit tree shows the sequence and relationship of commits using symbols: "
            "* = a commit, |\\ = branch split, |/ = branch merge, |* = commit on a side branch."
        ),
        "few_shot_template": """
Example {example_num}:
Commit Tree:
{graph_commits}
Release Note:
{cleaned_release_note}
""",
        "current_template": """
Commit Tree:
{graph_commits}
Release Note:
""",
        "get_input": lambda ex: ex["graph_commits"],
    },
    "rn": {
        "test_file": "../rn/test.json",
        "prompt_file": "../rn/prompt.json",
        "output_dir": "output/rn/",
        "prompt_instruction": (
            "Based on the given commit messages, generate a release note summarizing the key changes including bug fixes, new features, enhancements and other changes."
        ),
        "few_shot_template": """
Example {example_num}:
Commit Messages:
{commits}
Release Note:
{cleaned_release_note}
""",
        "current_template": """
Commit Messages:
{commits}
Release Note:
""",
        "get_input": lambda ex: "\n".join([c["cleaned_msg"] for c in ex["commits"]]),
    },
    "diff_rn": {
        "test_file": "../diff_rn/test.json",
        "prompt_file": "../diff_rn/prompt.json",
        "output_dir": "output/diff_rn/",
        "prompt_instruction": (
            "Based on the given code diff including file modification and diff, generate a release note summarizing the key changes including bug fixes, new features, enhancements and other changes."
        ),
        "few_shot_template": """
Example {example_num}:
Mod & Diff:
{mod_diff}
Release Note:
{cleaned_release_note}
""",
        "current_template": """
Mod & Diff:
{mod_diff}
Release Note:
""",
        "get_input": lambda ex: get_mod_diff_text(ex),
    },
}

def get_mod_diff_text(example):
    mod_str = ""
    if "mod" in example and example["mod"]:
        for m in example["mod"]:
            status = m.get("status", "")
            files = ", ".join(m.get("filename", []))
            mod_str += f"Status: {status}\nFiles: {files}\n"
    diff_str = example.get("diff", "")
    return mod_str.strip() + "\n\n" + diff_str

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
    # Initialize tokenizer and compute model max length
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    max_tokens = get_model_max_length(model_name)
    safety_margin = 512  # Reserve tokens for model response

    # Start with the prompt instruction
    prompt = task_cfg["prompt_instruction"] + "\n\n"
    used_tokens = len(tokenizer.encode(prompt))

    # Append few-shot examples if provided
    if few_shot_examples:
        intro = f"I will provide you with {len(few_shot_examples)} example{'s' if len(few_shot_examples) > 1 else ''}:\n\n"
        prompt += intro
        used_tokens += len(tokenizer.encode(intro))
        for i, ex in enumerate(few_shot_examples):
            # Prepare example input based on task type
            if 'graph_commits' in ex:
                shot_input = ex['graph_commits'].strip()
            elif 'commits' in ex:
                shot_input = "\n".join([c['cleaned_msg'] for c in ex['commits']]).strip()
            else:
                shot_input = ex['mod_diff'].strip()

            # Fill in the few-shot template
            example_block = task_cfg["few_shot_template"].format(
                example_num=i+1,
                graph_commits=shot_input,
                commits=shot_input,
                mod_diff=shot_input,
                cleaned_release_note=ex['cleaned_release_note'].strip()
            ) + "\n"
            block_tokens = len(tokenizer.encode(example_block))

            # Stop including examples if we're near the token limit
            if used_tokens + block_tokens > max_tokens - safety_margin:
                break

            prompt += example_block
            used_tokens += block_tokens

    # Construct the template for the current input
    current_block = task_cfg["current_template"].format(
        graph_commits=current_input.strip(),
        commits=current_input.strip(),
        mod_diff=current_input.strip()
    )
    block_tokens = len(tokenizer.encode(current_block))

    # If the current block exceeds the limit, truncate line by line
    if used_tokens + block_tokens > max_tokens - safety_margin:
        lines = current_input.strip().split('\n')
        truncated_lines = []
        for line in lines:
            trial_input = '\n'.join(truncated_lines + [line])
            trial_block = task_cfg["current_template"].format(
                graph_commits=trial_input,
                commits=trial_input,
                mod_diff=trial_input
            )
            if used_tokens + len(tokenizer.encode(trial_block)) > max_tokens - safety_margin:
                break
            truncated_lines.append(line)

        # Add an ellipsis marker to indicate truncation
        truncated_text = '\n'.join(truncated_lines) + '\n... (truncated)'
        current_block = task_cfg["current_template"].format(
            graph_commits=truncated_text,
            commits=truncated_text,
            mod_diff=truncated_text
        )

    # Combine and return the final prompt
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
        # ROUGE-L
        scores = rouge.get_scores(" ".join(hypothesis), " ".join(reference))
        rouge_l = round(scores[0]["rouge-l"]["f"] * 100, 4)
        # BLEU-4
        try:
            bleu_score = nltk.translate.bleu_score.corpus_bleu(
                [[reference]], [hypothesis],
                smoothing_function=smoothing,
                weights=(0.25, 0.25, 0.25, 0.25)
            )
            bleu = round(bleu_score * 100, 4)
        except Exception:
            bleu = 0.0
        # METEOR
        try:
            meteor = meteor_score([reference], hypothesis)
            meteor = round(meteor * 100, 4)
        except Exception:
            meteor = 0.0
        return rouge_l, bleu, meteor

    # 多线程并行
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(single_eval, zip(preds, refs)))

    for rouge_l, bleu, meteor in results:
        score_dict["rouge-l"].append(rouge_l)
        score_dict["bleu-4"].append(bleu)
        score_dict["meteor"].append(meteor)

    # 计算平均值
    return {k: round(sum(v) / len(v), 4) for k, v in score_dict.items()}

def run_task(task_name, args, llm):
    task_cfg = TASKS[task_name]
    logger.info(f"\n========== [START] Task: {task_name} | Model: {args.model} | Few-shot: {args.num_few_shot} ==========")
    with open(task_cfg["test_file"], "r") as f:
        test_data = json.load(f)
        if args.test_size is not None:
            test_data = test_data[:args.test_size]
            logger.info(f"Processing first {args.test_size} test examples for {task_name}")
    with open(task_cfg["prompt_file"], "r") as f:
        prompt_data = json.load(f)
    few_shot_examples = []
    if args.num_few_shot > 0:
        if task_name == "diff_rn":
            for ex in prompt_data[:args.num_few_shot]:
                ex_mod_diff = get_mod_diff_text(ex)
                few_shot_examples.append({
                    "mod_diff": ex_mod_diff,
                    "cleaned_release_note": ex["cleaned_release_note"]
                })
        else:
            few_shot_examples = prompt_data[:args.num_few_shot]
        logger.info(f"Using {len(few_shot_examples)} few-shot examples for {task_name}")
    predictions = []
    total_examples = len(test_data)
    logger.info(f"Starting inference on {total_examples} examples for {task_name}")
    # 动态获取tokenizer和max_tokens
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
    # 分批推理
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
                "cleaned_release_note": example["cleaned_release_note"],
                "prediction": prediction
            })
    os.makedirs(task_cfg["output_dir"], exist_ok=True)
    output_file = os.path.join(
        task_cfg["output_dir"],
        f"{args.model.split('/')[-1]}_few_shot_{args.num_few_shot}.json"
    )
    with open(output_file, "w") as f:
        json.dump(predictions, f, indent=2)
    logger.info(f"[Task: {task_name}] Saved predictions to {output_file}")
    # 评估
    preds = [p["prediction"] for p in predictions]
    refs = [p["cleaned_release_note"] for p in predictions]
    metrics = compute_metrics(preds, refs)
    logger.info(f"[Task: {task_name}] Evaluation Metrics:")
    logger.info(f"[Task: {task_name}] BLEU-4: {metrics['bleu-4']:.4f}")
    logger.info(f"[Task: {task_name}] ROUGE-L: {metrics['rouge-l']:.4f}")
    logger.info(f"[Task: {task_name}] METEOR: {metrics['meteor']:.4f}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(task_cfg["output_dir"], "evaluation_results.csv")
    header = ["timestamp", "model", "num_few_shot", "bleu-4", "rouge-l", "meteor"]
    row = {
        "timestamp": timestamp,
        "model": args.model.split("/")[-1],
        "num_few_shot": args.num_few_shot,
        "bleu-4": metrics["bleu-4"],
        "rouge-l": metrics["rouge-l"],
        "meteor": metrics["meteor"]
    }
    write_header = not os.path.exists(csv_path)
    with open(csv_path, mode='a', newline='', encoding='utf-8') as f:
        import csv
        writer = csv.DictWriter(f, fieldnames=header)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    logger.info(f"[Task: {task_name}] Saved evaluation results to {csv_path}")
    logger.info(f"========== [END] Task: {task_name} | Model: {args.model} | Few-shot: {args.num_few_shot} ==========")

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
    parser.add_argument("--only_task", type=str, default=None, choices=["graph_rn", "rn", "diff_rn"], help="Only run the specified task")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    num_gpus = torch.cuda.device_count()
    # 动态获取模型最大长度
    model_max_len = get_model_max_length(args.model)
    logger.info(f"[Global] Using model: {args.model}, max_model_len: {model_max_len}")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=max(1, num_gpus),
        max_num_batched_tokens=model_max_len,
        max_model_len=model_max_len,
        gpu_memory_utilization=0.9,  # 提高显存利用率
        swap_space=4,  # 启用显存交换，单位GB
        block_size=16,  # 优化KV cache
        trust_remote_code=True,
        tokenizer_mode="auto",
        enforce_eager=True,
    )

    # 新增：diff_rn 只跑0-shot，其余任务全shot
    # for task_name in ["graph_rn", "rn", "diff_rn"]:
    #     if task_name == "diff_rn":
    #         shots = [0]
    #     else:
    #         shots = [0, 1, 2, 4]
    #     for few_shot in shots:
    #         args.num_few_shot = few_shot
    #         run_task(task_name, args, llm)
    for task_name in ["diff_rn"]:
        args.num_few_shot = 0
        run_task(task_name, args, llm)
        
if __name__ == "__main__":
    main() 