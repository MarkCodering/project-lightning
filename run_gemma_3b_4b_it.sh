
#!/usr/bin/env bash
set -euo pipefail

# Suppress HuggingFace tokenizers fork/parallelism warning
export TOKENIZERS_PARALLELISM=false

# Gemma-3 4B IT: SFT (math + MBPP) then GRPO with math + MBPP + HumanEval (for experimentation)
# NOTE: Including HumanEval in RL (or SFT) contaminates the benchmark; use only for ablations.

python train_quantum_reasoner.py \
  --base_model google/gemma-3-4b-it \
  --output_dir ./runs/gemma-3-4b-qrl-v2-qk-prm \
  --skip_baseline_eval \
  --enable_qk_prm \
  --rl_steps 300 \
  --sft_include_mbpp \
  --sft_mbpp_limit 1000 \
  --rl_include_mbpp \
  --rl_mbpp_limit 500 \
  --flash_attn \
  --compile_models \
  --disable_sft_gc \
  --use_8bit_optim

echo "Run complete. Outputs in ./runs/gemma-3-4b-qrl-v2-qk-prm"