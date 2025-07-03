#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:h100:4
#SBATCH --mem=720G
#SBATCH --output=log/infer_all_%j.log

source ~/.bashrc
conda activate rsum

export CUDA_VISIBLE_DEVICES=0,1,2,3

MODELS=(
    "Qwen/Qwen2.5-7B-Instruct"
    "mistralai/Mistral-Small-Instruct-2409"
    "Qwen/Qwen2.5-32B-Instruct"
    "meta-llama/Llama-3.3-70B-Instruct"
    "Qwen/Qwen2.5-72B-Instruct"
)
FEWSHOTS=(0 1 2 4)
MAX_BATCHED_TOKENS=4096
MAX_TOKENS=256
TEMPERATURE=0.0
TOP_P=1.0
TEST_SIZE=1000

mkdir -p log
mkdir -p error_logs

echo "Starting all-models inference at $(date)"

for MODEL in "${MODELS[@]}"; do
  LOGFILE=log/infer_all_$(echo $MODEL | tr '/' '_' | tr ':' '_')_alltasks_$(date +%Y%m%d_%H%M%S).log
  python3 infer_all_tasks.py \
    --model "$MODEL" \
    --max_batched_tokens $MAX_BATCHED_TOKENS \
    --max_tokens $MAX_TOKENS \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --test_size $TEST_SIZE \
    > "$LOGFILE" 2>&1
  if [ $? -ne 0 ]; then
    echo "Error: Failed for $MODEL, see $LOGFILE"
    head -n 20 "$LOGFILE"
    echo "-----------------------------"
  fi
  sleep 5
  echo "Finished $MODEL at $(date)"
done

echo "All-models inference finished at $(date)" 
