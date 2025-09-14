python train_quantum_reasoner.py \
  --base_model google/gemma-3-4b-it \
  --output_dir ./runs/gemma-3-4b-qrl-v2-qk-prm \
  --skip_baseline_eval \
  --enable_qk_prm \
  --rl_steps 300 \
  \
  # Include all datasets for SFT (math + code). Note: HumanEval in SFT contaminates its benchmark.
  --sft_include_mbpp \
  --sft_mbpp_limit 1000 \
  \
  # Include all datasets for RL (math + code) with practical limits
  --rl_include_mbpp \
  --rl_include_humaneval \
  --rl_mbpp_limit 500 \
  --rl_humaneval_limit 164