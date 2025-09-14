python train_quantum_reasoner.py \
  --base_model google/gemma-3-4b-it \
  --output_dir ./runs/gemma-3-4b-qrl-v2-qk-prm \
  --skip_baseline_eval \
  --enable_qk_prm \
  --rl_steps 300