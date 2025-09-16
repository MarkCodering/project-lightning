# -*- coding: utf-8 -*-
"""
Clean SFT -> GRPO training + clean evaluation (no test-set contamination).

- SFT: GSM8K(train) + Hendrycks MATH(train) [+ optional MBPP train/val only]
- RL (GRPO): same train-only pools (never HumanEval, never MBPP test)
- Eval: GSM8K(test), MBPP(test), HumanEval(test)  [HumanEval is eval-only by default]

Fixes:
- Explicit use_cache=False before enabling gradient checkpointing (no HF warnings)
- prepare_model_for_kbit_training BEFORE LoRA wrapping
- Real gradient-norm logging via callback (no more 0.0 lies)
- Safer LR/warmup/max_grad_norm to stabilize early steps
"""

import os, re, json, time, argparse, tempfile, subprocess, warnings, copy, glob, math
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    GenerationConfig,
    DataCollatorForLanguageModeling,
    TrainerCallback,
)
from transformers.utils import logging as hf_logging
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig, GRPOTrainer, GRPOConfig
from tqdm.auto import tqdm

hf_logging.set_verbosity_error()

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("medium")
except Exception:
    pass

# --------------------------- small utils ---------------------------

def _norm(x: str) -> str:
    return re.sub(r"\s+", " ", x).strip()

def extract_gsm8k_final(s: str) -> str:
    m = re.search(r"####\s*([^\n]+)", s)
    return _norm(m.group(1) if m else s)

def extract_math_final(solution: str) -> str:
    m = re.search(r"\\boxed\{([^}]*)\}", solution)
    if m:
        return _norm(m.group(1))
    tail = re.findall(r"[-+]?\d*\.?\d+(?:/\d+)?", solution)
    return _norm(tail[-1]) if tail else _norm(solution)

def equal_numeric(a: str, b: str, tol=1e-6) -> bool:
    a, b = _norm(a), _norm(b)
    if a == b:
        return True

    def parse_num(s):
        s = s.strip()
        if "/" in s and not re.match(r"^-?\d+\.\d+$", s):
            try:
                n, d = s.split("/")
                return float(n) / float(d)
            except Exception:
                pass
        try:
            return float(s)
        except Exception:
            return None

    ax, bx = parse_num(a), parse_num(b)
    return (ax is not None and bx is not None and abs(ax - bx) <= tol)

def parse_final_from_completion(txt: str) -> str:
    m = re.search(r"<answer>\s*(.*?)</answer>", txt, re.S | re.I)
    if m:
        return _norm(m.group(1))
    tail = re.findall(r"[-+]?\d*\.?\d+(?:/\d+)?", txt)
    return _norm(tail[-1]) if tail else _norm(txt[-64:])

# ---------------------- stopping criteria -------------------------

class StopOnSubstrings(StoppingCriteria):
    def __init__(self, tokenizer: "AutoTokenizer", substrings: List[str]):
        super().__init__()
        self.tokenizer = tokenizer
        self.sequences: List[List[int]] = []
        for s in substrings:
            try:
                ids = tokenizer.encode(s, add_special_tokens=False)
            except Exception:
                ids = []
            if ids:
                self.sequences.append(ids)

    def _endswith_ids(self, row_ids: torch.Tensor, seq_ids: List[int]) -> bool:
        if row_ids.size(0) < len(seq_ids):
            return False
        tail = row_ids[-len(seq_ids):]
        return torch.equal(tail, torch.tensor(seq_ids, device=tail.device))

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor, **kwargs) -> bool:
        if not self.sequences:
            return False
        for row in input_ids:
            for seq in self.sequences:
                if self._endswith_ids(row, seq):
                    return True
        return False

def build_stop_criteria(tok: "AutoTokenizer") -> StoppingCriteriaList:
    stops = [
        "</answer>", "\n</answer>", "</answer>\n", "\n\n</answer>",
        "\n```", "\n\n```",
    ]
    return StoppingCriteriaList([StopOnSubstrings(tok, stops)])

# -------------------------- prompts --------------------------------

MATH_SYS = "You are a careful mathematician. Solve step by step in <think>...</think>, then put ONLY the final short answer inside <answer>...</answer>."
CODE_SYS = "You are a helpful coding assistant. Write correct, efficient Python code. Return ONLY code if tests will run."

def p_math(q: str) -> str:
    return f"{MATH_SYS}\n\n<task>\n{q.strip()}\n</task>"

def t_math(thought: str, final: str) -> str:
    return f"<think>\n{thought.strip()}\n</think>\n<answer>{final.strip()}</answer>"

def p_code(desc: str) -> str:
    return f"{CODE_SYS}\n\nProblem:\n{desc.strip()}\n\nWrite the solution."

# ---------------------- dataset loaders (CLEAN) --------------------

HENDRYCKS_CFGS = [
    "algebra","counting_and_probability","geometry","intermediate_algebra",
    "number_theory","prealgebra","precalculus",
]

def load_math_train() -> Dataset:
    gsm = load_dataset("openai/gsm8k", "main", split="train")
    gsm = gsm.map(lambda ex: {
        "source": "gsm8k",
        "prompt": p_math(ex["question"]),
        # Supervise on final answer only to avoid long CoT targets
        "target": f"<answer>{extract_gsm8k_final(ex['answer'])}</answer>",
        "final": extract_gsm8k_final(ex["answer"]),
    })
    math_parts = []
    for cfg in HENDRYCKS_CFGS:
        try:
            ds = load_dataset("EleutherAI/hendrycks_math", cfg, split="train")
            ds = ds.map(lambda ex: {
                "source": f"math/{cfg}",
                "prompt": p_math(ex["problem"]),
                # Supervise on final answer only to avoid long CoT targets
                "target": f"<answer>{extract_math_final(ex['solution'])}</answer>",
                "final": extract_math_final(ex["solution"]),
            })
            math_parts.append(ds)
        except Exception as e:
            warnings.warn(f"Skipping hendrycks_math '{cfg}': {e}")
    math_all = concatenate_datasets(math_parts) if math_parts else None
    return concatenate_datasets([gsm, math_all]) if math_all is not None else gsm

def _first_nonempty(d: dict, keys: List[str]) -> Optional[str]:
    for k in keys:
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return None

def load_mbpp_trainval(limit: Optional[int] = None) -> Optional[Dataset]:
    """
    Train-only MBPP: uses train + validation splits from sanitized MBPP if available.
    Never touches the official test set. Returns prompt/target pairs where reference code exists.
    """
    tried = [
        ("google-research-datasets/mbpp", "sanitized"),
        ("google-research-datasets/mbpp", None),
        ("evalplus/mbppplus", None),
        ("Muennighoff/mbpp", None),
    ]
    for repo, cfg in tried:
        try:
            # collect only non-test splits
            splits = []
            for split in ["train", "validation"]:
                try:
                    ds = load_dataset(repo, cfg, split=split)
                    splits.append(ds)
                except Exception:
                    continue
            if not splits:
                continue
            ds_all = concatenate_datasets(splits) if len(splits) > 1 else splits[0]

            def mapper(ex):
                desc = _first_nonempty(ex, ["text","prompt","description","task_description"]) or ""
                code = _first_nonempty(ex, ["code","code_string","code_solution","solution"]) or ""
                out = {"source":"mbpp_train","prompt": p_code(desc)}
                out["target"] = code.strip() if code.strip() else None
                return out

            ds_map = ds_all.map(mapper)
            ds_map = ds_map.filter(lambda ex: isinstance(ex.get("target"), str) and len(ex["target"].strip()) > 0)
            if limit is not None:
                ds_map = ds_map.select(range(min(limit, len(ds_map))))
            if len(ds_map) == 0:
                continue
            return ds_map
        except Exception:
            continue
    warnings.warn("MBPP train/val could not be constructed without test; skipping MBPP in training.")
    return None

def _map_mbpp_eval_columns(ds: Dataset) -> Dataset:
    def to_list_tests(x):
        if isinstance(x, list): return x
        if isinstance(x, str):  return [x]
        return []
    def map_row(ex):
        text = ex.get("text") or ex.get("prompt") or ex.get("description") or ex.get("task_description") or ""
        setup = ex.get("test_setup_code") or ex.get("setup_code") or ""
        tests = ex.get("test_list") or ex.get("tests") or ex.get("test") or []
        tests = to_list_tests(tests)
        return {"text": text, "test_setup_code": setup, "test_list": tests}
    keep = {"text","test_setup_code","test_list"}
    return ds.map(map_row, remove_columns=[c for c in ds.column_names if c not in keep])

def load_mbpp_test(n: Optional[int] = None) -> Dataset:
    # Always official test split for eval; never used in training here.
    choice = None
    for repo, cfg in [("google-research-datasets/mbpp","sanitized"),
                      ("google-research-datasets/mbpp", None),
                      ("evalplus/mbppplus", None)]:
        try:
            load_dataset(repo, cfg, split="test")
            choice = (repo, cfg); break
        except Exception:
            continue
    if choice is None:
        raise RuntimeError("Could not load MBPP test split for clean evaluation.")
    repo, cfg = choice
    ds = load_dataset(repo, cfg, split="test")
    ds = _map_mbpp_eval_columns(ds)
    if n is not None:
        ds = ds.select(range(min(n, len(ds))))
    return ds

def load_gsm8k_test(n: Optional[int] = None) -> Dataset:
    ds = load_dataset("openai/gsm8k", "main", split="test")
    if n is not None:
        ds = ds.select(range(min(n, len(ds))))
    return ds

def load_humaneval_test(n: Optional[int] = None) -> Dataset:
    ds = load_dataset("openai/openai_humaneval", split="test")
    if n is not None:
        ds = ds.select(range(min(n, len(ds))))
    return ds

# ---------------------- tiny sandbox for code ----------------------

def run_code_and_tests(code: str, setup: str, tests: List[str], timeout_s=6) -> Tuple[bool, str]:
    snippet = "\n".join([setup or "", code, "\n".join(tests)])
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(snippet); path = f.name
    try:
        proc = subprocess.run(
            ["python", path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=timeout_s
        )
        return proc.returncode == 0, (proc.stdout + "\n" + proc.stderr)
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    finally:
        try: os.remove(path)
        except Exception: pass

# ----------------- generation config helpers ----------------------

def safe_clone_gen_cfg(gc: Optional[GenerationConfig]) -> GenerationConfig:
    if gc is None:
        return GenerationConfig()
    try:
        return gc.clone()
    except Exception:
        if hasattr(gc, "to_dict"):
            return GenerationConfig(**gc.to_dict())
        return copy.deepcopy(gc)

def build_gen_cfg(model, do_sample: bool, top_p=None, top_k=None, temperature=None) -> GenerationConfig:
    gen = safe_clone_gen_cfg(getattr(model, "generation_config", None))
    if not hasattr(gen, "cache_implementation") or gen.cache_implementation is None:
        gen.cache_implementation = "hybrid"
    gen.do_sample = bool(do_sample)
    if do_sample:
        gen.top_p = float(top_p) if top_p is not None else None
        gen.top_k = int(top_k) if top_k is not None else None
        gen.temperature = float(temperature) if temperature is not None else None
    else:
        gen.top_p = None; gen.top_k = None; gen.temperature = None
    return gen

def finitemax_len(tok_max: int) -> bool:
    return tok_max is not None and tok_max != float("inf") and tok_max < 100_000

def encode_prompts(tok, prompts: List[str], device, max_len_prompt: int):
    tok_max = getattr(tok, "model_max_length", None)
    max_len = max_len_prompt if not finitemax_len(tok_max) else min(tok_max, max_len_prompt)
    return tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(device)

# ------------------------- EVALS (CLEAN) --------------------------

@torch.inference_mode()
def eval_gsm8k(model, tok, n=100, max_new_tokens=256,
               do_sample=False, top_p=None, top_k=None, temperature=None,
               batch_size: int = 8, eval_prompt_max_len: int = 2048):
    ds = load_gsm8k_test(n)
    questions = ds["question"]
    finals = [extract_gsm8k_final(x) for x in ds["answer"]]
    total = len(questions)
    device = model.device
    gen_cfg = build_gen_cfg(model, do_sample, top_p, top_k, temperature)
    stop_crit = build_stop_criteria(tok)

    correct = 0
    start_t = time.time()
    pbar = tqdm(total=total, desc="GSM8K eval", dynamic_ncols=True, smoothing=0.1)

    for i in range(0, total, batch_size):
        batch_q = questions[i:i+batch_size]
        batch_final = finals[i:i+batch_size]
        prompts = [p_math(q) for q in batch_q]
        inputs = encode_prompts(tok, prompts, device, max_len_prompt=eval_prompt_max_len)
        out = model.generate(**inputs, generation_config=gen_cfg, max_new_tokens=max_new_tokens, stopping_criteria=stop_crit)
        decoded = tok.batch_decode(out, skip_special_tokens=True)
        for j, full in enumerate(decoded):
            pred = parse_final_from_completion(full[len(prompts[j]):])
            if equal_numeric(pred, batch_final[j]):
                correct += 1
        done = min(i+batch_size, total)
        elapsed = max(time.time() - start_t, 1e-6)
        pbar.update(len(prompts))
        pbar.set_postfix(acc=f"{correct/done:.3f}", ex_s=f"{done/elapsed:.2f}")
    pbar.close()
    return correct / total

@torch.inference_mode()
def eval_mbpp(model, tok, n=50, max_new_tokens=200,
              do_sample=False, top_p=None, top_k=None, temperature=None,
              batch_size: int = 4, eval_prompt_max_len: int = 1536):
    ds = load_mbpp_test(n)
    descs = ds["text"]
    setups = ds["test_setup_code"] if "test_setup_code" in ds.column_names else [""] * len(ds)
    tests = ds["test_list"] if "test_list" in ds.column_names else ([[]] * len(ds))
    total = len(descs)
    device = model.device
    gen_cfg = build_gen_cfg(model, do_sample, top_p, top_k, temperature)
    stop_crit = build_stop_criteria(tok)
    passed = 0
    start_t = time.time()
    pbar = tqdm(total=total, desc="MBPP eval", dynamic_ncols=True, smoothing=0.1)

    for i in range(0, total, batch_size):
        batch_desc = descs[i:i+batch_size]
        prompts = [p_code(d) for d in batch_desc]
        inputs = encode_prompts(tok, prompts, device, max_len_prompt=eval_prompt_max_len)
        out = model.generate(**inputs, generation_config=gen_cfg, max_new_tokens=max_new_tokens, stopping_criteria=stop_crit)
        decoded = tok.batch_decode(out, skip_special_tokens=True)
        for j, full in enumerate(decoded):
            txt = full[len(prompts[j]):]
            m = re.search(r"```(?:python)?\s*(.*?)```", txt, re.S|re.I)
            code = m.group(1) if m else txt
            ok, _ = run_code_and_tests(code, setups[i+j], tests[i+j])
            if ok:
                passed += 1
        done = min(i+batch_size, total)
        elapsed = max(time.time() - start_t, 1e-6)
        pbar.update(len(prompts))
        pbar.set_postfix(pass1=f"{passed/done:.3f}", ex_s=f"{done/elapsed:.2f}")
    pbar.close()
    return passed / total

@torch.inference_mode()
def eval_humaneval(model, tok, n=30, max_new_tokens=200,
                   do_sample=False, top_p=None, top_k=None, temperature=None,
                   batch_size: int = 4, eval_prompt_max_len: int = 1536):
    ds = load_humaneval_test(n)
    prompts_raw = ds["prompt"]
    tests_raw = ds["test"]
    total = len(prompts_raw)
    device = model.device
    gen_cfg = build_gen_cfg(model, do_sample, top_p, top_k, temperature)
    stop_crit = build_stop_criteria(tok)
    passed = 0
    start_t = time.time()
    pbar = tqdm(total=total, desc="HumanEval eval", dynamic_ncols=True, smoothing=0.1)

    for i in range(0, total, batch_size):
        batch_prompt = prompts_raw[i:i+batch_size]
        prompts = [p_code(p) for p in batch_prompt]
        inputs = encode_prompts(tok, prompts, device, max_len_prompt=eval_prompt_max_len)
        out = model.generate(**inputs, generation_config=gen_cfg, max_new_tokens=max_new_tokens, stopping_criteria=stop_crit)
        decoded = tok.batch_decode(out, skip_special_tokens=True)
        for j, full in enumerate(decoded):
            txt = full[len(prompts[j]):]
            m = re.search(r"```(?:python)?\s*(.*?)```", txt, re.S|re.I)
            code = m.group(1) if m else txt
            ok, _ = run_code_and_tests(code, "", [tests_raw[i+j]])
            if ok:
                passed += 1
        done = min(i+batch_size, total)
        elapsed = max(time.time() - start_t, 1e-6)
        pbar.update(len(prompts))
        pbar.set_postfix(pass1=f"{passed/done:.3f}", ex_s=f"{done/elapsed:.2f}")
    pbar.close()
    return passed / total

def eval_all(label, model, tok, outdir,
             do_sample=False, top_p=None, top_k=None, temperature=None,
             batch_size=8, eval_prompt_max_len=2048,
             eval_mbpp_tasks=50, eval_humaneval_tasks=30, eval_gsm8k_tasks=100):
    # switch to left padding for eval
    orig_side = tok.padding_side
    tok.padding_side = "left"

    metrics = {"label": label}
    metrics["gsm8k_acc"] = eval_gsm8k(model, tok, n=eval_gsm8k_tasks,
                                      do_sample=do_sample, top_p=top_p, top_k=top_k, temperature=temperature,
                                      batch_size=batch_size, eval_prompt_max_len=eval_prompt_max_len)
    metrics["mbpp_pass1"] = eval_mbpp(model, tok, n=eval_mbpp_tasks,
                                      do_sample=do_sample, top_p=top_p, top_k=top_k, temperature=temperature,
                                      batch_size=max(1, batch_size//2), eval_prompt_max_len=max(512, eval_prompt_max_len//2))
    metrics["humaneval_pass1"] = eval_humaneval(model, tok, n=eval_humaneval_tasks,
                                                do_sample=do_sample, top_p=top_p, top_k=top_k, temperature=temperature,
                                                batch_size=max(1, batch_size//2), eval_prompt_max_len=max(512, eval_prompt_max_len//2))

    path = os.path.join(outdir, f"metrics_{label}.json")
    with open(path, "w") as f: json.dump(metrics, f, indent=2)
    tok.padding_side = orig_side
    return metrics

# ------------------------ LoRA helpers ----------------------------

def find_lora_targets(model) -> list:
    candidates = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj",
                  "wq","wk","wv","wo","w1","w2","w3","proj","out_proj"]
    hits = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            base = name.split(".")[-1]
            if base in candidates or any(c in base for c in candidates):
                hits.add(base)
    if not hits:
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and (
                "attn" in name.lower() or "attention" in name.lower() or
                "mlp" in name.lower() or "feed_forward" in name.lower()
            ):
                hits.add(name.split(".")[-1])
    hits = sorted(hits)
    if not hits:
        raise RuntimeError("Could not auto-detect LoRA target modules.")
    return hits

def print_trainable_params(model):
    trainable, total = 0, 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    pct = 100.0 * trainable / max(1, total)
    print(f"[Params] trainable={trainable:,} / total={total:,} ({pct:.4f}%)")

def forcebf16(module: nn.Module):
    changed = 0
    for p in module.parameters():
        if p.dtype == torch.float32:
            p.data = p.data.to(torch.bfloat16); changed += 1
    for b in module.buffers():
        if b.dtype == torch.float32:
            b.data = b.data.to(torch.bfloat16); changed += 1
    if changed:
        print(f"[dtype] Forced {changed} tensors to bfloat16 for consistency.")

# -------------------- Gradient norm recorder ----------------------

class GradNormRecorder:
    def __init__(self):
        self.accum_sq = 0.0
        self.last = 0.0
        self._hooks = []
        self.calls = 0
        self.hooked_params = 0

    def _hook(self, grad: torch.Tensor):
        try:
            # Use float32 for accumulation stability; grad may be bf16
            self.accum_sq += float(grad.detach().to(torch.float32).pow(2).sum().item())
            self.calls += 1
        except Exception:
            pass
        return grad

    def register(self, model: nn.Module):
        # register on all trainable parameters
        for p in model.parameters():
            if p.requires_grad:
                try:
                    h = p.register_hook(self._hook)
                    self._hooks.append(h)
                    self.hooked_params += 1
                except Exception:
                    continue
    def register_from_optimizer(self, optimizer):
        try:
            for group in getattr(optimizer, 'param_groups', []):
                for p in group.get('params', []):
                    if isinstance(p, torch.Tensor) and hasattr(p, 'register_hook'):
                        try:
                            h = p.register_hook(self._hook)
                            self._hooks.append(h)
                            self.hooked_params += 1
                        except Exception:
                            continue
        except Exception:
            pass

    def pop(self) -> float:
        val = (self.accum_sq ** 0.5) if self.accum_sq > 0 else 0.0
        self.last = val
        self.accum_sq = 0.0
        self.calls = 0
        return val

    def peek(self) -> float:
        if self.accum_sq > 0:
            return (self.accum_sq ** 0.5)
        return self.last

# -------------------- SFT builder (CLEAN) -------------------------

class GradNormOverrideCallback(TrainerCallback):
    """Override HF's default grad_norm=0.0 with our computed value."""
    def __init__(self):
        self.last_grad_norm = 0.0
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        # Compute gradient norm at log time when we know trainer has run backward
        if logs is None:
            return

        # Prefer the model passed in kwargs to avoid attribute access issues
        model = kwargs.get("model")
        if model is None:
            trainer = kwargs.get("trainer")
            model = getattr(trainer, "model", None) if trainer is not None else None
        if model is None:
            return
            
        # Try recorder first
        rec: GradNormRecorder = getattr(model, "_grad_recorder", None)
        if rec is not None:
            gn = float(rec.peek())
        else:
            total_sq = 0.0
            counted = 0
            for name, p in model.named_parameters():
                if p.grad is not None and torch.is_floating_point(p.grad):
                    g = p.grad.data
                    total_sq += float(g.norm(2) ** 2)
                    counted += 1
            gn = (total_sq ** 0.5) if counted > 0 else 0.0

        if gn <= 0:
            prev = getattr(model, "_last_grad_norm", None)
            if prev is not None:
                gn = float(prev)
        if gn > 0:
            self.last_grad_norm = gn
            logs["grad_norm"] = self.last_grad_norm

    def on_train_begin(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        trainer = kwargs.get("trainer")
        if model is None and trainer is not None:
            model = getattr(trainer, "model", None)
        if model is None:
            return
        # Attach recorder after model is fully prepared/wrapped by accelerate
        if getattr(model, "_grad_recorder", None) is None:
            try:
                rec = GradNormRecorder()
                rec.register(model)
                # If nothing was hooked yet, try optimizer params too
                if getattr(rec, 'hooked_params', 0) == 0 and trainer is not None:
                    opt = getattr(trainer, 'optimizer', None)
                    if opt is not None:
                        rec.register_from_optimizer(opt)
                setattr(model, "_grad_recorder", rec)
                print(f"[Grad] Recorder hooks registered (on_train_begin). hooked_params={rec.hooked_params}")
            except Exception as e:
                print(f"[Grad] Recorder registration failed (on_train_begin): {e}")

    def on_step_begin(self, args, state, control, **kwargs):
        # Fallback: if recorder present but not hooked, try optimizer now
        model = kwargs.get("model")
        trainer = kwargs.get("trainer")
        if model is None:
            return
        rec = getattr(model, "_grad_recorder", None)
        if rec is not None and getattr(rec, 'hooked_params', 0) == 0 and trainer is not None:
            try:
                opt = getattr(trainer, 'optimizer', None)
                if opt is not None:
                    rec.register_from_optimizer(opt)
                    print(f"[Grad] Recorder late-registered on optimizer params. hooked_params={rec.hooked_params}")
            except Exception:
                pass

class GradNormCallback(TrainerCallback):
    def on_before_optimizer_step(self, args, state, control, optimizer=None, **kwargs):
        model = kwargs.get("model")
        if model is None:
            return
        try:
            max_gn = getattr(args, 'max_grad_norm', 1.0)
            if isinstance(max_gn, (int, float)) and max_gn > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_gn)
        except Exception:
            pass
    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        trainer = kwargs.get("trainer")
        if model is None:
            return
        # Prefer recorder if available (captures grads before zeroing), snapshot at step end
        rec: GradNormRecorder = getattr(model, "_grad_recorder", None)
        if rec is not None:
            gn = float(rec.pop())
        else:
            total_sq = 0.0
            counted = 0
            for p in model.parameters():
                if p.grad is not None and torch.is_floating_point(p.grad):
                    g = p.grad.data
                    total_sq += float(g.norm(2) ** 2)
                    counted += 1
            gn = (total_sq ** 0.5) if counted > 0 else 0.0
        # Optional: clip gradients to configured max_grad_norm for stability
        # Clipping is now handled in on_before_optimizer_step

        try:
            setattr(model, "_last_grad_norm", float(gn))
        except Exception:
            pass

        # One-time diagnostics at first step
        try:
            if int(state.global_step) == 1 and trainer is not None:
                rec = getattr(model, "_grad_recorder", None)
                opt = getattr(trainer, 'optimizer', None)
                n_groups = len(opt.param_groups) if opt is not None else 0
                trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                has_grad = sum(p.numel() for p in model.parameters() if getattr(p, 'grad', None) is not None)
                # labels coverage (if batch still available via trainer)
                label_cov = None
                try:
                    last_inputs = getattr(trainer, 'current_flos', None)  # placeholder; real inputs not available
                except Exception:
                    last_inputs = None
                print(f"[Diag] step=1 hooked_params={getattr(rec,'hooked_params',0)} hook_calls={getattr(rec,'calls',0)} trainable={trainable} has_grad={has_grad} opt_groups={n_groups}")
        except Exception:
            pass
        if trainer is not None:
            # Only log on configured logging steps to avoid spam
            try:
                log_every = max(1, getattr(args, "logging_steps", 5))
                first = bool(getattr(args, "logging_first_step", False))
            except Exception:
                log_every = 5
                first = False
            if state.global_step == 1 and first:
                trainer.log({"micro_grad_norm": gn, "grad_norm": gn})
            elif state.global_step % log_every == 0:
                trainer.log({"micro_grad_norm": gn, "grad_norm": gn})
        try:
            from tqdm.auto import tqdm as _tqdm
            rec = getattr(model, "_grad_recorder", None)
            rc = getattr(rec, 'calls', 0) if rec is not None else 0
            hp = getattr(rec, 'hooked_params', 0) if rec is not None else 0
            _tqdm.write(f"[GradNorm] step={state.global_step} micro_grad_norm={gn:.4f} (hook_calls={rc} hooked_params={hp})")
        except Exception:
            print(f"[GradNorm] step={state.global_step} micro_grad_norm={gn:.4f}")

def pretokenize_for_sft(raw_ds: Dataset, tok: "AutoTokenizer", max_len: int) -> Dataset:
    tok.model_max_length = max_len
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    use_chat = hasattr(tok, "apply_chat_template")

    def encode_example_chat(ex):
        p = ex["prompt"]; t = ex["target"]
        msgs_p = [{"role":"user","content": p}]
        msgs_full = [{"role":"user","content": p}, {"role":"assistant","content": t}]

        enc_p = tok.apply_chat_template(msgs_p, add_generation_prompt=True, tokenize=True)
        enc_f = tok.apply_chat_template(msgs_full, add_generation_prompt=False, tokenize=True)
        p_ids = enc_p["input_ids"] if isinstance(enc_p, dict) else enc_p
        f_ids = enc_f["input_ids"] if isinstance(enc_f, dict) else enc_f
        # Flatten if nested (single example sometimes returns [ids])
        if isinstance(p_ids, list) and len(p_ids) > 0 and isinstance(p_ids[0], list):
            p_ids = p_ids[0]
        if isinstance(f_ids, list) and len(f_ids) > 0 and isinstance(f_ids[0], list):
            f_ids = f_ids[0]
        # Target portion is the tail after the prompt portion
        t_ids = f_ids[len(p_ids):]

        # Truncate from the left to fit max_len, preferring to keep target
        if len(f_ids) > max_len:
            # number to drop
            drop = len(f_ids) - max_len
            f_ids = f_ids[drop:]
            # Prompt kept length adjusts accordingly
            kept_p = max(0, len(p_ids) - drop)
        else:
            kept_p = len(p_ids)
        # After truncation, labels are -100 for kept prompt tokens, then actual ids for the rest
        input_ids = f_ids
        labels = ([-100] * kept_p) + f_ids[kept_p:]
        attention_mask = [1] * len(input_ids)
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}

    def encode_example_basic(ex):
        p = ex["prompt"]; t = ex["target"]
        p_ids = tok(p, add_special_tokens=False)["input_ids"]
        t_ids = tok(t, add_special_tokens=False)["input_ids"]
        eos = [tok.eos_token_id] if tok.eos_token_id is not None else []
        max_target = max_len - len(eos)
        if len(t_ids) > max_target:
            t_ids = t_ids[:max_target]
        remain = max_len - len(t_ids) - len(eos)
        if remain < 0:
            remain = 0
        if len(p_ids) > remain:
            p_ids = p_ids[-remain:] if remain > 0 else []
        input_ids = p_ids + t_ids + eos
        labels = ([-100] * len(p_ids)) + t_ids + eos
        attention_mask = [1] * len(input_ids)
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}

    mapper = encode_example_chat if use_chat else encode_example_basic
    ds_tok = raw_ds.map(mapper, remove_columns=list(raw_ds.column_names))
    ds_tok.set_format(type="torch")
    return ds_tok

class DataCollatorKeepLabels:
    def __init__(self, tokenizer: AutoTokenizer, label_pad_token_id: int = -100):
        self.tok = tokenizer
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, features: List[dict]) -> dict:
        # Ensure lists for pad()
        def tolist(x):
            if isinstance(x, torch.Tensor):
                return x.tolist()
            return x
        batch_inputs = {
            "input_ids": [tolist(f["input_ids"]) for f in features],
            "attention_mask": [tolist(f["attention_mask"]) for f in features],
        }
        batch = self.tok.pad(batch_inputs, padding=True, return_tensors="pt")
        max_len = batch["input_ids"].size(1)
        labels = torch.full((len(features), max_len), self.label_pad_token_id, dtype=torch.long)
        for i, f in enumerate(features):
            li = tolist(f["labels"]) if "labels" in f else tolist(f["input_ids"])  # fallback
            L = min(len(li), max_len)
            labels[i, :L] = torch.tensor(li[:L], dtype=torch.long)
        batch["labels"] = labels
        return batch

def build_sft_trainer(base_model, output_dir, sft_steps=300, lr=2e-5,
                      per_device_train_batch_size=1, grad_acc=8, bf16=True, max_len=4096,
                      include_mbpp_train: bool = False, mbpp_limit: Optional[int] = None,
                      attn_impl: str = "eager",
                      compile_models: bool = False,
                      disable_sft_gc: bool = False,
                      use_8bit_optim: bool = False,
                      dataloader_workers: int = 4):
    tok = AutoTokenizer.from_pretrained(base_model, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    tok.model_max_length = max_len

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation=attn_impl,
    )
    forcebf16(model)

    # Fix 1: set use_cache=False BEFORE enabling GC
    model.config.use_cache = False

    # Enable GC (if desired)
    if not disable_sft_gc:
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except Exception:
            try: model.gradient_checkpointing_enable()
            except Exception: pass

    # Ensure inputs require grads (HF/PEFT + GC quirk)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        # Fallback for older versions
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # Fix 2: prepare for k-bit training BEFORE LoRA; let TRL inject LoRA via peft_config
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=not disable_sft_gc)

    targets = find_lora_targets(model)
    print(f"[LoRA] target_modules={targets}")
    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=targets
    )
    # Do NOT wrap here; pass peft_config to SFTTrainer to ensure optimizer picks LoRA params
    try:
        model.config.use_cache = False
    except Exception:
        pass

    # Generation config for SFT (deterministic)
    gc = safe_clone_gen_cfg(getattr(model, "generation_config", None))
    gc.do_sample = False; gc.top_p = None; gc.top_k = None; gc.temperature = None
    gc.cache_implementation = "hybrid"
    model.generation_config = gc

    raw = load_math_train()
    if include_mbpp_train:
        mbpp_ds = load_mbpp_trainval(limit=mbpp_limit)
        if mbpp_ds is not None and len(mbpp_ds) > 0:
            raw = concatenate_datasets([raw, mbpp_ds])
            print(f"[SFT] Added MBPP train/val examples: {len(mbpp_ds):,}")
        else:
            print("[SFT] MBPP train/val not available; continuing without it.")

    train_ds = pretokenize_for_sft(raw, tok, max_len=max_len)
    # Use a collator that preserves provided masked labels
    collator = DataCollatorKeepLabels(tok, label_pad_token_id=-100)
    optim_choice = "paged_adamw_8bit" if use_8bit_optim else "adamw_torch"

    args = SFTConfig(
        output_dir=os.path.join(output_dir, "sft"),
        max_steps=sft_steps,
        learning_rate=min(lr, 3.0e-6),             # tighter LR cap for stability
        warmup_steps=max(100, int(0.20 * sft_steps)),  # 20% warmup or at least 100 steps
        warmup_ratio=0.0,                          # disable ratio-based warmup
        lr_scheduler_type="cosine",
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=grad_acc,
        bf16=bf16,
        logging_steps=5,
        logging_first_step=True,
        disable_tqdm=False,
        report_to="none",
        save_steps=200,
        gradient_checkpointing=not disable_sft_gc,
        remove_unused_columns=False,
        group_by_length=True,
        optim=optim_choice,
        dataloader_num_workers=dataloader_workers,
        dataloader_pin_memory=True,
        max_grad_norm=0.2,                         # even tighter clip
        weight_decay=0.0,                          # disable weight decay for LoRA
        adam_beta1=0.9,
        adam_beta2=0.95,
    )

    trainer = SFTTrainer(
        model=model, args=args, train_dataset=train_ds,
        data_collator=collator, peft_config=lora,
    )
    
    # Debug: print optimizer param groups and initial learning rates
    def print_optimizer_lr(trainer):
        opt = getattr(trainer, 'optimizer', None)
        if opt is not None:
            for i, group in enumerate(opt.param_groups):
                print(f"[DEBUG] Optimizer group {i} lr={group['lr']}")
    print_optimizer_lr(trainer)
    try:
        print_trainable_params(trainer.model)
    except Exception:
        pass

    # Recorder hooks will be attached on on_train_begin after Accelerate wraps the model

    # Force initial LR to be nonzero for all param groups
    opt = getattr(trainer, 'optimizer', None)
    if opt is not None:
        for group in opt.param_groups:
            if group['lr'] == 0.0:
                group['lr'] = args.learning_rate if hasattr(args, 'learning_rate') else 2e-5
                print(f"[FIX] Set optimizer group lr to {group['lr']}")
    
    trainer.add_callback(GradNormOverrideCallback())
    trainer.add_callback(GradNormCallback())
    
    if compile_models:
        # Advice: leave this OFF for SFT unless you've verified stability.
        try:
            trainer.model = torch.compile(trainer.model, mode="reduce-overhead", fullgraph=False)
            print("[Perf] Compiled model with torch.compile")
        except Exception as e:
            print(f"[Perf] torch.compile skipped: {e}")

    return trainer, tok

# -------------------- GRPO builder (CLEAN) ------------------------

def build_grpo_trainer(base_model, sft_ckpt, output_dir, rl_steps=300,
                       per_device_train_batch_size=1, grad_acc=4, bf16=True,
                       group_size=4, max_len=4096,
                       rl_top_p: Optional[float] = 0.9,
                       rl_top_k: Optional[int]   = 40,
                       rl_temperature: float = 0.8,
                       max_completion_length: int = 256,
                       include_mbpp_train: bool = False,
                       mbpp_limit: Optional[int] = None,
                       attn_impl: str = "eager",
                       compile_models: bool = False,
                       use_8bit_optim: bool = False):

    tok = AutoTokenizer.from_pretrained(base_model, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)

    # base model for RL + (optional) SFT adapter
    base_model_for_rl = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation=attn_impl,
    )
    forcebf16(base_model_for_rl)

    # RL config: no GC, fast generation, keep cache on
    base_model_for_rl.config.use_cache = True
    base_model_for_rl = prepare_model_for_kbit_training(base_model_for_rl, use_gradient_checkpointing=False)
    forcebf16(base_model_for_rl)

    model = base_model_for_rl
    if sft_ckpt and os.path.isdir(sft_ckpt):
        cp = sft_ckpt
        if not os.path.basename(cp).startswith("checkpoint-"):
            cands = glob.glob(os.path.join(cp, "checkpoint-*"))
            if cands:
                cp = max(cands, key=os.path.getctime)
        print(f"[GRPO] Loading SFT adapter from: {cp}")
        model = PeftModel.from_pretrained(base_model_for_rl, cp, is_trainable=True)
    else:
        print("[GRPO] No SFT adapter provided; RL will start from base model LoRA-free (trainable heads only).")

    forcebf16(model)
    model.train()
    print_trainable_params(model)

    # Explicit sampling config for RL
    gc = safe_clone_gen_cfg(getattr(model, "generation_config", None))
    gc.cache_implementation = "hybrid"
    gc.do_sample = True
    gc.temperature = rl_temperature
    gc.top_p = rl_top_p
    gc.top_k = rl_top_k
    gc.pad_token_id = tok.eos_token_id
    gc.eos_token_id = tok.eos_token_id
    gc.repetition_penalty = 1.05
    model.generation_config = gc
    print(f"[GRPO] Sampling: temp={rl_temperature}, top_p={rl_top_p}, top_k={rl_top_k}")

    # Build train datasets (train-only, CLEAN)
    mt = load_math_train()
    # remove 'final' if present (not needed for RL reward below)
    if "final" in mt.column_names:
        mt = mt.remove_columns(["final"])
    math_train = mt.map(lambda ex: {"task":"math"})

    code_train = None
    if include_mbpp_train:
        mbpp_tv = load_mbpp_trainval(limit=mbpp_limit)
        if mbpp_tv is not None and len(mbpp_tv) > 0:
            def map_code_row(ex):
                return {"prompt": ex["prompt"], "target": ex["target"], "task":"code"}
            code_train = mbpp_tv.map(map_code_row, remove_columns=[c for c in mbpp_tv.column_names if c not in ["prompt","target","task"]])
            print(f"[GRPO] MBPP RL train examples (train/val only): {len(code_train):,}")
        else:
            print("[GRPO] MBPP train/val not available; RL will be math-only.")

    mix = concatenate_datasets([math_train, code_train]).shuffle(seed=42) if code_train is not None else math_train.shuffle(seed=42)

    # Reward functions (no test-set leakage)
    def reward_math(prompts=None, completions=None, **kwargs):
        rewards = []
        for comp in completions:
            has = 1.0 if re.search(r"<answer>.*?</answer>", comp, re.S) else 0.0
            penalty = min(len(comp) / 2000.0, 1.0)  # length penalty
            r = 0.5 * has + 0.5 * (1.0 - penalty)
            rewards.append(float(r))
        return rewards

    def reward_code(prompts=None, completions=None, **kwargs):
        rewards = []
        for comp in completions:
            m = re.search(r"```(?:python)?\s*(.*?)```", comp, re.S|re.I)
            code = m.group(1) if m else comp
            score = 0.0
            if "def " in code: score += 0.5
            if "return " in code: score += 0.3
            if len(code) > 0: score += 0.2
            rewards.append(float(score))
        return rewards

    def mixed_reward(prompts=None, completions=None, task=None, **kw):
        out = []
        for i,t in enumerate(task):
            if t == "math":
                out.extend(reward_math(prompts=[prompts[i]], completions=[completions[i]]))
            else:
                out.extend(reward_code(prompts=[prompts[i]], completions=[completions[i]]))
        return out

    generation_kwargs = {
        "do_sample": True,
        "temperature": rl_temperature,
        "top_p": rl_top_p,
        "top_k": rl_top_k,
        "pad_token_id": tok.eos_token_id,
        "eos_token_id": tok.eos_token_id,
        "repetition_penalty": 1.05,
        "cache_implementation": "hybrid",
    }

    args = GRPOConfig(
        output_dir=os.path.join(output_dir, "grpo"),
        max_steps=rl_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=grad_acc,
        bf16=bf16,
        logging_steps=20, logging_first_step=True,
        disable_tqdm=False, report_to="none",
        num_generations=group_size,
        generation_batch_size=group_size,
        temperature=rl_temperature, top_p=rl_top_p, top_k=rl_top_k,
        max_prompt_length=max_len,
        max_completion_length=max_completion_length,
        generation_kwargs=generation_kwargs,
        gradient_checkpointing=False,
        remove_unused_columns=False,
        optim=("paged_adamw_8bit" if use_8bit_optim else "adamw_torch"),
    )

    trainer = GRPOTrainer(
        model=model, processing_class=tok, reward_funcs=mixed_reward,
        train_dataset=mix, args=args,
    )

    if compile_models:
        try:
            trainer.model = torch.compile(trainer.model, mode="reduce-overhead", fullgraph=False)
            print("[Perf] Compiled RL model with torch.compile")
        except Exception as e:
            print(f"[Perf] RL torch.compile skipped: {e}")

    return trainer, tok

# ------------------------------ main --------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", type=str, default="google/gemma-3-4b-it")
    ap.add_argument("--output_dir", type=str, default="./runs/gemma3-clean")
    ap.add_argument("--sft_steps", type=int, default=300)
    ap.add_argument("--rl_steps", type=int, default=300)
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--grad_acc", type=int, default=8)
    ap.add_argument("--group_size", type=int, default=4)
    ap.add_argument("--max_len", type=int, default=4096)
    ap.add_argument("--bf16", action="store_true"); ap.add_argument("--no_bf16", dest="bf16", action="store_false"); ap.set_defaults(bf16=True)

    # Train data toggles (CLEAN defaults)
    ap.add_argument("--include_mbpp_train", action="store_true", help="Use MBPP train/val in SFT and RL (never test).")
    ap.add_argument("--mbpp_limit", type=int, default=None)

    # Attention / perf
    ap.add_argument("--attn", type=str, choices=["eager","sdpa","flash"], default="eager")
    ap.add_argument("--compile_models", action="store_true")
    ap.add_argument("--disable_sft_gc", action="store_true")
    ap.add_argument("--use_8bit_optim", action="store_true")

    # Eval sampling & batching
    ap.add_argument("--eval_sample", action="store_true")
    ap.add_argument("--eval_top_p", type=float, default=0.9)
    ap.add_argument("--eval_top_k", type=int,   default=40)
    ap.add_argument("--eval_temperature", type=float, default=0.7)
    ap.add_argument("--eval_batch_size", type=int, default=8)
    ap.add_argument("--eval_prompt_max_len", type=int, default=2048)
    ap.add_argument("--eval_mbpp_tasks", type=int, default=50)
    ap.add_argument("--eval_humaneval_tasks", type=int, default=30)
    ap.add_argument("--eval_gsm8k_tasks", type=int, default=100)

    # Flow control
    ap.add_argument("--skip_baseline_eval", action="store_true")
    ap.add_argument("--skip_sft", action="store_true")
    ap.add_argument("--sft_ckpt", type=str, default=None, help="Use an existing SFT folder or checkpoint-*")
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--grpo_ckpt", type=str, default=None)

    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    attn_map = {"eager":"eager","sdpa":"sdpa","flash":"flash_attention_2"}

    baseline = None
    if not args.skip_baseline_eval:
        tok0 = AutoTokenizer.from_pretrained(args.base_model, use_fast=True, trust_remote_code=True)
        if tok0.pad_token is None: tok0.pad_token = tok0.eos_token
        tok0.padding_side = "right"

        bnb0 = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                                  bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
        base = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb0,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation=attn_map.get(args.attn, "eager"),
        )
        forcebf16(base)
        base.config.use_cache = False
        gc0 = safe_clone_gen_cfg(getattr(base, "generation_config", None))
        gc0.do_sample = False; gc0.top_p = None; gc0.top_k = None; gc0.temperature = None
        gc0.cache_implementation = "hybrid"
        base.generation_config = gc0

        baseline = eval_all(
            "baseline", base, tok0, args.output_dir,
            do_sample=args.eval_sample,
            top_p=args.eval_top_p if args.eval_sample else None,
            top_k=args.eval_top_k if args.eval_sample else None,
            temperature=args.eval_temperature if args.eval_sample else None,
            batch_size=args.eval_batch_size,
            eval_prompt_max_len=args.eval_prompt_max_len,
            eval_mbpp_tasks=args.eval_mbpp_tasks,
            eval_humaneval_tasks=args.eval_humaneval_tasks,
            eval_gsm8k_tasks=args.eval_gsm8k_tasks,
        )
        print("[Baseline]", json.dumps(baseline, indent=2))
    else:
        print("[Baseline] Skipped baseline evaluation.")

    if args.skip_sft and not args.sft_ckpt:
        print("[Main] SFT skipped without a checkpoint; RL will start from base model.")
    elif not args.skip_sft:
        sft_trainer, tok_sft = build_sft_trainer(
            base_model=args.base_model, output_dir=args.output_dir,
            sft_steps=args.sft_steps, grad_acc=args.grad_acc, bf16=args.bf16, max_len=args.max_len,
            include_mbpp_train=args.include_mbpp_train, mbpp_limit=args.mbpp_limit,
            attn_impl=attn_map.get(args.attn, "eager"),
            compile_models=args.compile_models,     # consider off until stable
            disable_sft_gc=args.disable_sft_gc,
            use_8bit_optim=args.use_8bit_optim,
        )
        sft_trainer.train()
        args.sft_ckpt = sft_trainer.args.output_dir
        print(f"[SFT] Done. ckpt={args.sft_ckpt}")

    if args.eval_only:
        adapter_path = args.grpo_ckpt or args.sft_ckpt
        if not adapter_path:
            raise SystemExit("--eval_only requires --grpo_ckpt or --sft_ckpt")
        print(f"[EvalOnly] Loading adapter from: {adapter_path}")

        tok_eval = AutoTokenizer.from_pretrained(args.base_model, use_fast=True, trust_remote_code=True)
        if tok_eval.pad_token is None: tok_eval.pad_token = tok_eval.eos_token
        tok_eval.padding_side = "right"

        bnb_eval = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                                      bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
        base_eval = AutoModelForCausalLM.from_pretrained(
            args.base_model, torch_dtype=torch.bfloat16, quantization_config=bnb_eval,
            device_map="auto", trust_remote_code=True, attn_implementation=attn_map.get(args.attn, "eager")
        )
        forcebf16(base_eval)

        cp = adapter_path
        if os.path.isdir(cp) and not os.path.basename(cp).startswith("checkpoint-"):
            cands = glob.glob(os.path.join(cp, "checkpoint-*"))
            if cands: cp = max(cands, key=os.path.getctime)

        try:
            model_eval = PeftModel.from_pretrained(base_eval, cp, is_trainable=False)
            print(f"[EvalOnly] Adapter loaded: {cp}")
        except Exception as e:
            print(f"[EvalOnly] Failed to load adapter at {cp}: {e}. Using base model only.")
            model_eval = base_eval

        metrics_trained = eval_all(
            "trained", model_eval, tok_eval, args.output_dir,
            do_sample=args.eval_sample,
            top_p=args.eval_top_p if args.eval_sample else None,
            top_k=args.eval_top_k if args.eval_sample else None,
            temperature=args.eval_temperature if args.eval_sample else None,
            batch_size=args.eval_batch_size,
            eval_prompt_max_len=args.eval_prompt_max_len,
            eval_mbpp_tasks=args.eval_mbpp_tasks,
            eval_humaneval_tasks=args.eval_humaneval_tasks,
            eval_gsm8k_tasks=args.eval_gsm8k_tasks,
        )
        print("[EvalOnly Metrics]", json.dumps(metrics_trained, indent=2))
        return

    # GRPO (clean)
    grpo_trainer, tok_rl = build_grpo_trainer(
        base_model=args.base_model, sft_ckpt=args.sft_ckpt, output_dir=args.output_dir,
        rl_steps=args.rl_steps, per_device_train_batch_size=args.per_device_train_batch_size,
        grad_acc=max(1, args.grad_acc//2), group_size=args.group_size, bf16=args.bf16, max_len=args.max_len,
        include_mbpp_train=args.include_mbpp_train, mbpp_limit=args.mbpp_limit,
        rl_top_p=args.eval_top_p, rl_top_k=args.eval_top_k, rl_temperature=args.eval_temperature,
        max_completion_length=256,
        attn_impl=attn_map.get(args.attn, "eager"),
        compile_models=args.compile_models,
        use_8bit_optim=args.use_8bit_optim,
    )
    grpo_trainer.train()

    # Post-train evals (clean)
    trained = eval_all(
        "trained", grpo_trainer.model, tok_rl, args.output_dir,
        do_sample=args.eval_sample,
        top_p=args.eval_top_p if args.eval_sample else None,
        top_k=args.eval_top_k if args.eval_sample else None,
        temperature=args.eval_temperature if args.eval_sample else None,
        batch_size=args.eval_batch_size,
        eval_prompt_max_len=args.eval_prompt_max_len,
        eval_mbpp_tasks=args.eval_mbpp_tasks,
        eval_humaneval_tasks=args.eval_humaneval_tasks,
        eval_gsm8k_tasks=args.eval_gsm8k_tasks,
    )
    print("[Trained]", json.dumps(trained, indent=2))

    print(f"Saved metrics to:\n  {os.path.join(args.output_dir,'metrics_baseline.json') if not args.skip_baseline_eval else '(baseline skipped)'}\n  {os.path.join(args.output_dir,'metrics_trained.json')}")
    print(f"SFT ckpt:  {args.sft_ckpt}\nGRPO ckpt: {grpo_trainer.args.output_dir}")

if __name__ == "__main__":
    main()
