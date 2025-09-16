#!/usr/bin/env bash
set -euo pipefail
export TOKENIZERS_PARALLELISM=false

python train_quantum_reasoner.py \
  --base_model google/gemma-3-4b-it \
  --output_dir ./runs/gemma3-qrl-clean \
  --skip_baseline_eval \
  --sft_steps 300 \
  --rl_steps 300 \
  --include_mbpp_train \        # optional; uses MBPP train/val ONLY (safe)
  --mbpp_limit 2000 \           # optional cap
  --eval_sample \               # enable sampling during eval if you want
  --eval_top_p 0.9 \
  --eval_top_k 40 \
  --eval_temperature 0.7 \
  --attn eager \                # change to flash if you have flash-attn
  --compile_models \
  --use_8bit_optim
