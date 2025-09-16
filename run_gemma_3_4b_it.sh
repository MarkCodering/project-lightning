#!/usr/bin/env bash
# Clean SFT -> GRPO run for Gemma 3 4B IT (no test-set contamination)
# Usage:
#   bash run_gemma_3_4b_it.sh            # full train (SFT -> GRPO) + post-train evals
#   bash run_gemma_3_4b_it.sh eval-only  # eval an existing adapter (set SFT_CKPT or GRPO_CKPT below)

set -euo pipefail

### ---------- knobs you might tweak ----------
BASE_MODEL="microsoft/Phi-4-mini-reasoning"
OUTDIR="./runs/phi4-reasoning-mini-qrl"
SFT_STEPS=300
RL_STEPS=300

# Keep MBPP safe: train/val only (never test)
INCLUDE_MBPP_TRAIN=true
MBPP_LIMIT=2000

# Eval settings
EVAL_SAMPLE=true
EVAL_TOP_P=0.9
EVAL_TOP_K=40
EVAL_T=0.7
EVAL_BS=8
EVAL_PROMPT_MAX_LEN=2048
EVAL_MBPP_TASKS=50
EVAL_HE_TASKS=30
EVAL_GSM_TASKS=100

# Performance toggles
ATTN_IMPL="eager"          # eager | sdpa | flash
USE_8BIT_OPTIM=true
COMPILE_MODELS=true       # recommend OFF during SFT until stable

# If you want eval-only, set one of these and pass "eval-only" as the first arg
SFT_CKPT=""                # e.g., ./runs/gemma3-clean/sft/checkpoint-300
GRPO_CKPT=""               # e.g., ./runs/gemma3-clean/grpo/checkpoint-300
### -------------------------------------------

# Env hygiene
export TOKENIZERS_PARALLELISM=false
export HF_HUB_ENABLE_HF_TRANSFER=1
export CUDA_LAUNCH_BLOCKING=0

# Optional: make bfloat16 the default matmul (helps Ampere+)
export PYTORCH_ASSUME_FINITES=true

# Sanity checks
if [[ ! -f "train_quantum_reasoner.py" ]]; then
  echo "ERROR: train_quantum_reasoner.py not found in $(pwd). Put the script here or adjust the path."
  exit 1
fi

mkdir -p "$OUTDIR"

# Build common arg list
COMMON_ARGS=( 
  --base_model "$BASE_MODEL"
  --output_dir "$OUTDIR"
  --rl_steps "$RL_STEPS"
  --grad_acc 24
  --max_len 1536
  --attn "$ATTN_IMPL"
  --eval_top_p "$EVAL_TOP_P"
  --eval_top_k "$EVAL_TOP_K"
  --eval_temperature "$EVAL_T"
  --eval_batch_size "$EVAL_BS"
  --eval_prompt_max_len "$EVAL_PROMPT_MAX_LEN"
  --eval_mbpp_tasks "$EVAL_MBPP_TASKS"
  --eval_humaneval_tasks "$EVAL_HE_TASKS"
  --eval_gsm8k_tasks "$EVAL_GSM_TASKS"
  --use_8bit_optim
)

# Optional toggles
$EVAL_SAMPLE && COMMON_ARGS+=( --eval_sample ) || true
$COMPILE_MODELS && COMMON_ARGS+=( --compile_models ) || true

# Train-time data toggle (safe MBPP: train/val only)
$INCLUDE_MBPP_TRAIN && COMMON_ARGS+=( --include_mbpp_train ) || true
[[ -n "${MBPP_LIMIT}" ]] && COMMON_ARGS+=( --mbpp_limit "$MBPP_LIMIT" )

# Route: eval-only vs full train
MODE="${1:-train}"

if [[ "$MODE" == "eval-only" ]]; then
  if [[ -z "$SFT_CKPT" && -z "$GRPO_CKPT" ]]; then
    echo "ERROR: eval-only requires SFT_CKPT or GRPO_CKPT set in the script."
    exit 1
  fi
  echo "[RUN] EVAL-ONLY"
  python train_quantum_reasoner.py \
    "${COMMON_ARGS[@]}" \
    --eval_only \
    ${SFT_CKPT:+--sft_ckpt "$SFT_CKPT"} \
    ${GRPO_CKPT:+--grpo_ckpt "$GRPO_CKPT"}

else
  echo "[RUN] TRAIN (GRPO) + POST-TRAIN EVALS"
  # keep compile OFF for SFT for stability; if COMPILE_MODELS=true it's passed and the script will try it
  python train_quantum_reasoner.py \
    "${COMMON_ARGS[@]}" \
    $( $INCLUDE_MBPP_TRAIN && echo --include_mbpp_train ) \
    ${MBPP_LIMIT:+--mbpp_limit "$MBPP_LIMIT"}
fi

echo "Done. Outputs in: $OUTDIR"
