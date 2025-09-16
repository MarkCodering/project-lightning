import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from train_quantum_reasoner import eval_all, safe_clone_gen_cfg, forcebf16

BASE_MODEL = os.environ.get("BASE_MODEL", "microsoft/Phi-4-mini-reasoning")
OUTDIR = os.environ.get("OUTDIR", "./runs/quick-baseline")

os.makedirs(OUTDIR, exist_ok=True)

attn_impl = os.environ.get("ATTN_IMPL", "eager")
attn_map = {"eager":"eager","sdpa":"sdpa","flash":"flash_attention_2"}

print(f"[QuickEval] Loading base model: {BASE_MODEL}")

tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

a2 = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    quantization_config=a2,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation=attn_map.get(attn_impl, "eager"),
)
forcebf16(model)
model.config.use_cache = False

gc = safe_clone_gen_cfg(getattr(model, "generation_config", None))
do_sample = os.environ.get("EVAL_SAMPLE", "0") in ("1","true","True")
gc.do_sample = do_sample
if do_sample:
    gc.top_p = float(os.environ.get("EVAL_TOP_P", 0.9))
    gc.top_k = int(os.environ.get("EVAL_TOP_K", 40))
    gc.temperature = float(os.environ.get("EVAL_T", 0.7))
else:
    for k in ["top_p","top_k","temperature"]:
        setattr(gc, k, None)
setattr(gc, "cache_implementation", "hybrid")
model.generation_config = gc

metrics = eval_all(
    "baseline", model, tok, OUTDIR,
    do_sample=do_sample,
    top_p=gc.top_p if do_sample else None,
    top_k=gc.top_k if do_sample else None,
    temperature=gc.temperature if do_sample else None,
    batch_size=4,
    eval_prompt_max_len=1024,
    eval_mbpp_tasks=4,
    eval_humaneval_tasks=4,
    eval_gsm8k_tasks=10,
)
print(json.dumps(metrics, indent=2))
