python train_quantum_reasoner.py \
  --base_model google/gemma-3-4b-it \
  --output_dir ./runs/gemma-3-4b-qrl \
  --skip_sft \
  --sft_ckpt ./runs/gemma-3-4b-qrl/sft \
  --skip_baseline_eval \
  --enable_qk_prm \