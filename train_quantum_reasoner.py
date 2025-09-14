# -*- coding: utf-8 -*-
"""
Gemma-3-270M reasoner: SFT -> GRPO (+ Quantum A/B), with fast batched eval,
clean progress bars, top-k/top-p controls, version-safe GenerationConfig,
robust MBPP/HumanEval loaders (no dataset scripts), explicit eval truncation,
TRL-version-safe SFT (pre-tokenized), and Hendrycks MATH config fix.

PATCHES (this version):
- Force attn_implementation="eager" on all model loads (Gemma-3 recommendation).
- SFT: use_cache=False + gradient_checkpointing=True (memory-friendly training).
- GRPO: use_cache=True + gradient_checkpointing=False (fast generation).
- **Critical**: In GRPO, call prepare_model_for_kbit_training on the BASE model
  *before* loading the LoRA adapter, and load adapter with is_trainable=True.
- Explicitly unfreeze lora_* params after loading; print trainable counts.
- Right-side padding; cleaner generation_config handling.
"""

import os, re, json, math, time, argparse, tempfile, subprocess, warnings, copy, glob
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
)
from transformers.utils import logging as hf_logging
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig, GRPOTrainer, GRPOConfig
from tqdm.auto import tqdm

# --------- clean logs & enable faster matmul ----------
hf_logging.set_verbosity_warning()
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("medium")
except Exception:
    pass

# ===================== Version-safe GenerationConfig clone =====================

def safe_clone_gen_cfg(gc: Optional[GenerationConfig]) -> GenerationConfig:
    if gc is None:
        return GenerationConfig()
    try:
        return gc.clone()
    except Exception:
        if hasattr(gc, "to_dict"):
            return GenerationConfig(**gc.to_dict())
        return copy.deepcopy(gc)

# ===================== Utilities: parsing/normalizing =====================

def _norm(x: str) -> str:
    return re.sub(r"\s+", " ", x).strip()

def extract_gsm8k_final(s: str) -> str:
    m = re.search(r"####\s*([^\n]+)", s)
    return _norm(m.group(1) if m else s)

def extract_math_final(solution: str) -> str:
    m = re.search(r"\\boxed\{([^}]*)\}", solution)
    if m: return _norm(m.group(1))
    tail = re.findall(r"[-+]?\d*\.?\d+(?:/\d+)?", solution)
    return _norm(tail[-1]) if tail else _norm(solution)

def equal_numeric(a: str, b: str, tol=1e-6) -> bool:
    if _norm(a) == _norm(b): return True
    def parse_num(s):
        s = s.strip()
        if "/" in s and not re.match(r"^-?\d+\.\d+$", s):
            try:
                n, d = s.split("/")
                return float(n)/float(d)
            except: pass
        try: return float(s)
        except: return None
    ax, bx = parse_num(a), parse_num(b)
    return (ax is not None and bx is not None and abs(ax-bx) <= tol)

def parse_final_from_completion(txt: str) -> str:
    m = re.search(r"<answer>\s*(.*?)</answer>", txt, re.S|re.I)
    if m: return _norm(m.group(1))
    tail = re.findall(r"[-+]?\d*\.?\d+(?:/\d+)?", txt)
    return _norm(tail[-1]) if tail else _norm(txt[-64:])

# ===================== Prompts =====================

MATH_SYS = "You are a careful mathematician. Solve step by step in <think>...</think>, then put ONLY the final short answer inside <answer>...</answer>."
CODE_SYS = "You are a helpful coding assistant. Write correct, efficient Python code. Return ONLY code if tests will run."

def p_math(q: str) -> str:
    return f"{MATH_SYS}\n\n<task>\n{q.strip()}\n</task>"

def t_math(thought: str, final: str) -> str:
    return f"<think>\n{thought.strip()}\n</think>\n<answer>{final.strip()}</answer>"

def p_code(desc: str) -> str:
    return f"{CODE_SYS}\n\nProblem:\n{desc.strip()}\n\nWrite the solution."

# ===================== Dataset loaders (NO script execution) =====================

HENDRYCKS_CFGS = [
    "algebra","counting_and_probability","geometry","intermediate_algebra",
    "number_theory","prealgebra","precalculus",
]

def load_math_sft():
    gsm = load_dataset("openai/gsm8k", "main", split="train")
    gsm = gsm.map(lambda ex: {
        "source": "gsm8k",
        "prompt": p_math(ex["question"]),
        "target": t_math(ex["answer"], extract_gsm8k_final(ex["answer"])),
        "final": extract_gsm8k_final(ex["answer"]),
        "teacher_think": ex["answer"],
    })
    math_parts = []
    for cfg in HENDRYCKS_CFGS:
        try:
            ds = load_dataset("EleutherAI/hendrycks_math", cfg, split="train")
            ds = ds.map(lambda ex: {
                "source": f"math/{cfg}",
                "prompt": p_math(ex["problem"]),
                "target": t_math(ex["solution"], extract_math_final(ex["solution"])),
                "final": extract_math_final(ex["solution"]),
                "teacher_think": ex["solution"],
            })
            math_parts.append(ds)
        except Exception as e:
            warnings.warn(f"Skipping hendrycks_math config '{cfg}': {e}")
    math_all = concatenate_datasets(math_parts) if math_parts else None
    return concatenate_datasets([gsm, math_all]) if math_all is not None else gsm

def _try_load_mbpp_variant() -> Optional[Tuple[str, Optional[str]]]:
    for repo, cfg in [("google-research-datasets/mbpp", None),
                      ("google-research-datasets/mbpp", "sanitized"),
                      ("evalplus/mbppplus", None)]:
        try:
            load_dataset(repo, cfg, split="test")
            return repo, cfg
        except Exception:
            continue
    return None

def _map_mbpp_columns(ds: Dataset) -> Dataset:
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
    choice = _try_load_mbpp_variant()
    if choice is None:
        raise RuntimeError("Could not load MBPP without dataset scripts. Try --skip_mbpp_eval.")
    repo, cfg = choice
    ds = load_dataset(repo, cfg, split="test")
    ds = _map_mbpp_columns(ds)
    if n is not None:
        ds = ds.select(range(min(n, len(ds))))
    return ds

def load_humaneval_test(n: Optional[int] = None) -> Dataset:
    ds = load_dataset("openai/openai_humaneval", split="test")
    if n is not None:
        ds = ds.select(range(min(n, len(ds))))
    return ds

# ===================== Code sandbox (demo; not secure) =====================

def run_code_and_tests(code: str, setup: str, tests: List[str], timeout_s=4) -> Tuple[bool, str]:
    snippet = "\n".join([setup or "", code, "\n".join(tests)])
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(snippet); path = f.name
    try:
        proc = subprocess.run(["python", path], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True, timeout=timeout_s)
        return proc.returncode == 0, proc.stderr
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    finally:
        try: os.remove(path)
        except: pass

# ===================== Generation config builder =====================

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

# ===================== Tokenization helpers =====================

def _finite_max_len(tok_max: int) -> bool:
    return tok_max is not None and tok_max != float("inf") and tok_max < 100_000

def encode_prompts(tok, prompts: List[str], device, max_len_prompt: int):
    tok_max = getattr(tok, "model_max_length", None)
    max_len = max_len_prompt if not _finite_max_len(tok_max) else min(tok_max, max_len_prompt)
    return tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(device)

# ===================== Batched evals =====================

@torch.inference_mode()
def eval_gsm8k(model, tok, n=100, max_new_tokens=256,
               do_sample=False, top_p=None, top_k=None, temperature=None,
               batch_size: int = 8, eval_prompt_max_len: int = 2048):
    ds = load_dataset("openai/gsm8k", "main", split="test").select(range(n))
    questions = ds["question"]
    finals = [extract_gsm8k_final(x) for x in ds["answer"]]
    total = len(questions)
    device = model.device
    gen_cfg = build_gen_cfg(model, do_sample, top_p, top_k, temperature)

    correct = 0
    start_t = time.time()
    pbar = tqdm(total=total, desc="GSM8K eval", dynamic_ncols=True, smoothing=0.1)
    for i in range(0, total, batch_size):
        batch_q = questions[i:i+batch_size]
        batch_final = finals[i:i+batch_size]
        prompts = [p_math(q) for q in batch_q]
        inputs = encode_prompts(tok, prompts, device, max_len_prompt=eval_prompt_max_len)
        out = model.generate(**inputs, generation_config=gen_cfg, max_new_tokens=max_new_tokens)
        decoded = tok.batch_decode(out, skip_special_tokens=True)
        for j, full in enumerate(decoded):
            pred = parse_final_from_completion(full[len(prompts[j]):])
            if equal_numeric(pred, batch_final[j]): correct += 1
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
    cols = set(ds.column_names)
    setups = ds["test_setup_code"] if "test_setup_code" in cols else [""] * len(ds)
    tests = ds["test_list"] if "test_list" in cols else ([[]] * len(ds))

    total = len(descs)
    device = model.device
    gen_cfg = build_gen_cfg(model, do_sample, top_p, top_k, temperature)

    passed = 0
    start_t = time.time()
    pbar = tqdm(total=total, desc="MBPP eval", dynamic_ncols=True, smoothing=0.1)
    for i in range(0, total, batch_size):
        batch_desc = descs[i:i+batch_size]
        prompts = [p_code(d) for d in batch_desc]
        inputs = encode_prompts(tok, prompts, device, max_len_prompt=eval_prompt_max_len)
        out = model.generate(**inputs, generation_config=gen_cfg, max_new_tokens=max_new_tokens)
        decoded = tok.batch_decode(out, skip_special_tokens=True)
        for j, full in enumerate(decoded):
            txt = full[len(prompts[j]):]
            m = re.search(r"```(?:python)?\s*(.*?)```", txt, re.S|re.I)
            code = m.group(1) if m else txt
            ok, _ = run_code_and_tests(code, setups[i+j], tests[i+j])
            if ok: passed += 1
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

    passed = 0
    start_t = time.time()
    pbar = tqdm(total=total, desc="HumanEval eval", dynamic_ncols=True, smoothing=0.1)
    for i in range(0, total, batch_size):
        batch_prompt = prompts_raw[i:i+batch_size]
        prompts = [p_code(p) for p in batch_prompt]
        inputs = encode_prompts(tok, prompts, device, max_len_prompt=eval_prompt_max_len)
        out = model.generate(**inputs, generation_config=gen_cfg, max_new_tokens=max_new_tokens)
        decoded = tok.batch_decode(out, skip_special_tokens=True)
        for j, full in enumerate(decoded):
            txt = full[len(prompts[j]):]
            m = re.search(r"```(?:python)?\s*(.*?)```", txt, re.S|re.I)
            code = m.group(1) if m else txt
            ok, _ = run_code_and_tests(code, "", [tests_raw[i+j]])
            if ok: passed += 1
        done = min(i+batch_size, total)
        elapsed = max(time.time() - start_t, 1e-6)
        pbar.update(len(prompts))
        pbar.set_postfix(pass1=f"{passed/done:.3f}", ex_s=f"{done/elapsed:.2f}")
    pbar.close()
    return passed / total

def eval_all(label, model, tok, outdir,
             do_sample=False, top_p=None, top_k=None, temperature=None,
             batch_size=8, eval_prompt_max_len=2048):
    # For decoder-only generation correctness, ensure left padding during eval
    orig_side = tok.padding_side
    tok.padding_side = "left"
    metrics = {
        "label": label,
        "gsm8k_acc": eval_gsm8k(model, tok, n=100, do_sample=do_sample, top_p=top_p, top_k=top_k, temperature=temperature, batch_size=batch_size, eval_prompt_max_len=eval_prompt_max_len),
        "mbpp_pass1": eval_mbpp(model, tok, n=50, do_sample=do_sample, top_p=top_p, top_k=top_k, temperature=temperature, batch_size=max(1, batch_size//2), eval_prompt_max_len=max(512, eval_prompt_max_len//2)),
        "humaneval_pass1": eval_humaneval(model, tok, n=30, do_sample=do_sample, top_p=top_p, top_k=top_k, temperature=temperature, batch_size=max(1, batch_size//2), eval_prompt_max_len=max(512, eval_prompt_max_len//2)),
    }
    path = os.path.join(outdir, f"metrics_{label}.json")
    with open(path, "w") as f: json.dump(metrics, f, indent=2)
    tok.padding_side = orig_side
    return metrics

def print_compare(baseline, trained):
    head = ["Benchmark", "Baseline", "Trained", "Δ"]
    rows = []
    for k,label in [("gsm8k_acc","GSM8K acc@1"),
                    ("mbpp_pass1","MBPP pass@1"),
                    ("humaneval_pass1","HumanEval pass@1")]:
        b = baseline[k]; t = trained[k]; d = t - b
        rows.append([label, f"{b:.3f}", f"{t:.3f}", f"{d:+.3f}"])
    colw = [max(len(x) for x in col) for col in zip(head, *rows)]
    def fmt(row): return " | ".join(x.ljust(w) for x,w in zip(row, colw))
    print("\n" + fmt(head))
    print("-+-".join("-"*w for w in colw))
    for r in rows: print(fmt(r))
    print()

# ===================== LoRA helpers =====================

def find_lora_targets(model) -> list:
    candidates = [
        "q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj",
        "wq","wk","wv","wo","w1","w2","w3","proj",
    ]
    hits = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            base = name.split(".")[-1]
            if any(base == c for c in candidates) or any(c in base for c in candidates):
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

# --- dtype helpers ---------------------------------------------------------
def _force_bf16(module: nn.Module):
    """Force all float32 params/buffers to bfloat16 (used to avoid Float/BFloat16 matmul).

    BitsAndBytes' prepare_model_for_kbit_training keeps LayerNorms in fp32 for stability,
    but Gemma3 eager attention currently expects uniform dtype for Q,K,V to avoid a
    RuntimeError (expected scalar type Float but found BFloat16). We accept the tiny
    potential stability trade-off to guarantee consistent dtypes during GRPO sampling.
    """
    changed = 0
    for p in module.parameters():
        if p.dtype == torch.float32:
            p.data = p.data.to(torch.bfloat16)
            changed += 1
    for b in module.buffers():
        if b.dtype == torch.float32:
            b.data = b.data.to(torch.bfloat16)
            changed += 1
    if changed:
        print(f"[dtype] Forced {changed} tensors to bfloat16 for consistency.")

# ===================== SFT builder =====================

def _pretokenize_for_sft(raw_ds: Dataset, tok: AutoTokenizer, max_len: int) -> Dataset:
    def to_text(batch):
        texts = [f"{p}\n\n{t}" for p, t in zip(batch["prompt"], batch["target"])]
        return {"text": texts}
    ds_text = raw_ds.map(to_text, batched=True, remove_columns=[c for c in raw_ds.column_names if c not in []])

    tok.model_max_length = max_len
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    def tok_map(batch):
        enc = tok(batch["text"], truncation=True, max_length=max_len, padding="max_length")
        labels = [ids.copy() for ids in enc["input_ids"]]
        enc["labels"] = labels
        return enc

    ds_tok = ds_text.map(tok_map, batched=True, remove_columns=["text"])
    ds_tok.set_format(type="torch")
    return ds_tok

def build_sft_trainer(base_model, output_dir, sft_steps=300, lr=2e-5,
                      per_device_train_batch_size=1, grad_acc=8, bf16=True, max_len=4096):
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
        attn_implementation="eager",
    )
    _force_bf16(model)
    model.config.use_cache = False  # SFT: checkpointing + no cache

    gc = safe_clone_gen_cfg(getattr(model, "generation_config", None))
    gc.do_sample = False; gc.top_p = None; gc.top_k = None; gc.temperature = None
    gc.cache_implementation = "hybrid"
    model.generation_config = gc

    try:
        model.gradient_checkpointing_enable()
    except Exception:
        pass

    targets = find_lora_targets(model)
    print(f"[LoRA] target_modules={targets}")

    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=targets
    )

    raw = load_math_sft()
    train_ds = _pretokenize_for_sft(raw, tok, max_len=max_len)
    collator = DataCollatorForLanguageModeling(tok, mlm=False)

    args = SFTConfig(
        output_dir=os.path.join(output_dir, "sft"),
        max_steps=sft_steps, learning_rate=lr,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=grad_acc, bf16=bf16,
        logging_steps=5, logging_first_step=True,
        disable_tqdm=False, report_to="none",
        save_steps=200,
        lr_scheduler_type="cosine", warmup_ratio=0.03,
        gradient_checkpointing=True,
    )

    trainer = SFTTrainer(
        model=model, args=args, train_dataset=train_ds,
        data_collator=collator, peft_config=lora,
    )

    print_trainable_params(trainer.model)
    return trainer, tok

# ===================== Quantum A (optional) =====================

try:
    import pennylane as qml
except Exception:
    qml = None
    warnings.warn("PennyLane not available; Quantum Bandit disabled.")

class QuantumBandit(nn.Module):
    def __init__(self, n_features: int, n_qubits: int = 6, n_layers: int = 2, shots: Optional[int] = None):
        super().__init__()
        if qml is None:
            raise RuntimeError("PennyLane not installed.")
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.pre = nn.Linear(n_features, n_qubits)
        self.dev = qml.device("default.qubit", wires=n_qubits, shots=shots)
        @qml.qnode(self.dev, interface="torch", diff_method="best")
        def circuit(x, weights):
            xpad = qml.math.pad(x, (0, max(0, n_qubits - x.shape[0])))[:n_qubits]
            for i in range(n_qubits):
                qml.RX(xpad[i], wires=i)
            for l in range(n_layers):
                for i in range(n_qubits):
                    qml.Rot(weights[l,i,0], weights[l,i,1], weights[l,i,2], wires=i)
                for i in range(0, n_qubits-1, 2): qml.CZ(wires=[i, i+1])
                for i in range(1, n_qubits-1, 2): qml.CZ(wires=[i, i+1])
            return [qml.expval(qml.PauliZ(0))]
        self.circuit = circuit
        self.weights = nn.Parameter(torch.randn(n_layers, n_qubits, 3) * 0.1)
    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        z = torch.tanh(self.pre(feats))
        outs = []
        for i in range(z.size(0)):
            val = self.circuit(z[i], self.weights)[0]
            outs.append(val)
        logits = torch.stack(outs)
        return torch.sigmoid(2.0 * logits)

def build_features_for_bandit(completions: List[str]) -> torch.Tensor:
    feats = []
    for txt in completions:
        think = re.findall(r"<think>(.*?)</think>", txt, re.S)
        think_text = "\n".join(think) if think else ""
        think_lines = think_text.count("\n") + 1 if think_text else 0
        ans_m = re.search(r"<answer>(.*?)</answer>", txt, re.S)
        ans_len = 0 if not ans_m else len(ans_m.group(1))
        code_blocks = len(re.findall(r"```", txt))
        digits = sum(c.isdigit() for c in txt)
        length = len(txt)
        feats.append([think_lines, ans_len, code_blocks, digits, length, 1.0])
    return torch.tensor(feats, dtype=torch.float32)

# ===================== Quantum B (optional) =====================

def embed_texts(texts: List[str], dim: int = 384) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        embs = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        return np.asarray(embs, dtype=np.float32)
    except Exception:
        D = 256
        vecs = np.zeros((len(texts), D), dtype=np.float32)
        for i,t in enumerate(texts):
            t = t.lower()
            for j in range(len(t)-2):
                tri = t[j:j+3]
                vecs[i, hash(tri) % D] += 1.0
            n = np.linalg.norm(vecs[i])+1e-8
            vecs[i] /= n
        return vecs

def qkernel_matrix(X: np.ndarray, Y: np.ndarray, reps=1) -> np.ndarray:
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    Yn = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-8)
    cos = Xn @ Yn.T
    return np.cos((math.pi/2.0) * (1 - cos)) ** reps

class QuantumKernelPRM:
    def __init__(self, lam=1e-3, reps=1):
        self.lam = lam; self.reps = reps
        self.X = None; self.alpha = None
    def fit(self, X: np.ndarray, y: np.ndarray):
        K = qkernel_matrix(X, X, reps=self.reps)
        n = K.shape[0]
        self.alpha = np.linalg.solve(K + self.lam * np.eye(n), y)
        self.X = X; return self
    def predict(self, Xq: np.ndarray) -> np.ndarray:
        Kq = qkernel_matrix(Xq, self.X, reps=self.reps)
        return Kq @ self.alpha

def split_steps_from_think(think_text: str) -> List[str]:
    think_text = think_text.strip()
    if not think_text: return []
    parts = [p.strip() for p in think_text.split("\n") if p.strip()]
    if len(parts) < 2:
        parts = re.split(r"(?<=[\.\!\?;])\s+", think_text)
        parts = [p.strip() for p in parts if p.strip()]
    return parts

def build_qkprm_from_teacher(math_sft, max_steps=50000, lam=1e-2, reps=1):
    pos_steps = []
    for ex in math_sft:
        t = ex.get("teacher_think","")
        t = re.sub(r"</?think>", "", t, flags=re.I)
        pos_steps.extend(split_steps_from_think(t))
        if len(pos_steps) >= max_steps: break
    pos_steps = pos_steps[:max_steps]
    neg_steps = []
    for ex in math_sft.shuffle(seed=13):
        s = ex.get("prompt","")
        body = re.sub(r".*?<task>\s*", "", s, flags=re.S)
        body = re.sub(r"\s*</task>.*", "", body, flags=re.S)
        if body.strip():
            neg_steps.append(body.strip())
        if len(neg_steps) >= len(pos_steps): break
    X_pos = embed_texts(pos_steps)
    X_neg = embed_texts(neg_steps[:len(pos_steps)])
    X = np.vstack([X_pos, X_neg])
    y = np.concatenate([np.ones(len(X_pos)), np.zeros(len(X_neg))]).astype(np.float32)
    return QuantumKernelPRM(lam=lam, reps=reps).fit(X, y)

def process_reward_for_completion(prm: QuantumKernelPRM, completion_text: str) -> float:
    think = re.findall(r"<think>(.*?)</think>", completion_text, re.S)
    think_text = "\n".join(think) if think else ""
    steps = split_steps_from_think(think_text)
    if not steps: return 0.0
    X = embed_texts(steps)
    scores = prm.predict(X)
    return float(np.mean(np.clip(scores, 0.0, 1.0)))

# ===================== GRPO builder (FIXED trainables) =====================

def build_grpo_trainer(base_model, sft_ckpt, output_dir, rl_steps=300,
                       per_device_train_batch_size=1, grad_acc=4, bf16=True,
                       group_size=4, max_len=4096,
                       enable_q_bandit=False, bandit_alpha=0.2,
                       enable_qk_prm=False, prm_weight=0.3,
                       rl_top_p: Optional[float] = 0.9,
                       rl_top_k: Optional[int]   = 40,
                       rl_temperature: float = 0.8,
                       max_completion_length: int = 256):
    tok = AutoTokenizer.from_pretrained(base_model, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)

    # 1) Load BASE model (eager, cache ON for fast generation)
    base_model_for_rl = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    _force_bf16(base_model_for_rl)
    base_model_for_rl.config.use_cache = True

    # 2) Prepare BASE for 4-bit training (NO gradient checkpointing in RL)
    base_model_for_rl = prepare_model_for_kbit_training(
        base_model_for_rl,
        use_gradient_checkpointing=False
    )
    _force_bf16(base_model_for_rl)

    # 3) Load LoRA adapter with is_trainable=True (this sets requires_grad on LoRA)
    latest_checkpoint = None
    # Allow passing a direct checkpoint path OR a parent directory containing checkpoint-* dirs
    if os.path.isdir(sft_ckpt):
        if os.path.basename(sft_ckpt).startswith("checkpoint-"):
            latest_checkpoint = sft_ckpt
        else:
            checkpoints = glob.glob(os.path.join(sft_ckpt, "checkpoint-*"))
            if checkpoints:
                latest_checkpoint = max(checkpoints, key=os.path.getctime)
    if latest_checkpoint:
        sft_model = PeftModel.from_pretrained(
            base_model_for_rl,
            latest_checkpoint,
            is_trainable=True
        )
        print(f"[GRPO] Loaded SFT adapter from: {latest_checkpoint}")
    else:
        print(f"[GRPO] No adapter checkpoints found (path: {sft_ckpt}); proceeding with base model only.")
        sft_model = base_model_for_rl

    # Ensure adapter params also consistent
    _force_bf16(sft_model)

    # 4) Paranoia: force-unfreeze any lora_* parameters and keep cache on
    lora_trainable = 0
    for n, p in sft_model.named_parameters():
        if "lora_" in n.lower():
            p.requires_grad = True
            lora_trainable += p.numel()
    try:
        sft_model.base_model.config.use_cache = True
    except Exception:
        pass

    # 5) Training mode + log flags
    sft_model.train()
    print(f"[GRPO Flags] use_cache={getattr(sft_model.config, 'use_cache', None)}; "
          f"has_gc_attr={hasattr(sft_model, 'is_gradient_checkpointing')}")
    print(f"[GRPO] LoRA trainable params (explicit): {lora_trainable:,}")
    print_trainable_params(sft_model)

    # Set explicit generation config to avoid warnings and improve completions
    gc = safe_clone_gen_cfg(getattr(sft_model, "generation_config", None))
    gc.cache_implementation = "hybrid"
    gc.do_sample = True
    gc.temperature = rl_temperature
    gc.top_p = rl_top_p  
    gc.top_k = rl_top_k
    gc.pad_token_id = tok.eos_token_id
    gc.eos_token_id = tok.eos_token_id
    gc.repetition_penalty = 1.05
    sft_model.generation_config = gc
    print(f"[GRPO] Set explicit generation config: temp={rl_temperature}, top_p={rl_top_p}, cache=hybrid")

    # Build GRPO dataset
    math_train = load_math_sft().remove_columns(["target"]).map(lambda ex: {"task":"math"})
    mbpp = load_mbpp_test(50)
    he = load_humaneval_test(30)

    def map_code_row(ex):
        d = ex.get("text") or ex.get("desc") or ex.get("prompt") or ""
        setup = ex.get("test_setup_code") if "test_setup_code" in ex else ""
        if "test_list" in ex: tlist = ex["test_list"]
        elif "test" in ex:    tlist = [ex["test"]]
        else:                 tlist = []
        return {"prompt": p_code(d), "task":"code", "test_setup_code": setup, "tests": tlist}

    mbpp_m = mbpp.map(map_code_row, remove_columns=[c for c in mbpp.column_names if c not in []])
    he_m   = he.map(map_code_row,   remove_columns=[c for c in he.column_names   if c not in []])

    code_train = concatenate_datasets([mbpp_m, he_m]).shuffle(seed=42)
    mix = concatenate_datasets([math_train, code_train]).shuffle(seed=42)

    # Optional quantum pieces
    qk_prm = None
    if enable_qk_prm:
        small = load_math_sft().select(range(min(2000, len(math_train))))
        qk_prm = build_qkprm_from_teacher(small, max_steps=20000, lam=1e-2, reps=1)

    q_bandit, q_bandit_opt, bandit_device = None, None, None
    if enable_q_bandit:
        if qml is None:
            raise RuntimeError("PennyLane not installed; cannot enable --enable_q_bandit.")
        q_bandit = QuantumBandit(n_features=6)
        bandit_device = "cuda" if torch.cuda.is_available() else "cpu"
        q_bandit.to(bandit_device)
        q_bandit_opt = torch.optim.Adam(q_bandit.parameters(), lr=1e-3)

    def reward_math(prompts=None, completions=None, final=None, **kwargs):
        rewards = []
        for comp, gt in zip(completions, final):
            pred = parse_final_from_completion(comp)
            r = 1.0 if equal_numeric(pred, gt) else 0.0
            if enable_qk_prm and qk_prm is not None:
                r += prm_weight * process_reward_for_completion(qk_prm, comp)
            rewards.append(r)
        if enable_q_bandit and q_bandit is not None:
            feats = build_features_for_bandit(completions).to(bandit_device)
            with torch.no_grad(): probs = q_bandit(feats)
            rewards = [float(r + bandit_alpha*(p.item()-0.5)) for r,p in zip(rewards, probs)]
            with torch.enable_grad():
                q_bandit_opt.zero_grad()
                probs2 = q_bandit(feats)
                baseline = float(np.mean([max(0.0, rr) for rr in rewards]))
                target = torch.tensor([1.0 if rr >= baseline else 0.0 for rr in rewards],
                                      dtype=torch.float32, device=probs2.device)
                F.binary_cross_entropy(probs2, target).backward(); q_bandit_opt.step()
        return rewards

    def reward_code(prompts=None, completions=None, test_setup_code=None, tests=None, **kwargs):
        rewards = []
        if test_setup_code is None: test_setup_code = [""] * len(completions)
        if tests is None: tests = [[] for _ in range(len(completions))]
        for comp, setup, tlist in zip(completions, test_setup_code, tests):
            m = re.search(r"```(?:python)?\s*(.*?)```", comp, re.S|re.I)
            code = m.group(1) if m else comp
            ok, _ = run_code_and_tests(code, setup, tlist)
            r = 1.0 if ok else 0.0
            if enable_qk_prm and qk_prm is not None:
                r += prm_weight * process_reward_for_completion(qk_prm, comp)
            rewards.append(r)
        if enable_q_bandit and q_bandit is not None:
            feats = build_features_for_bandit(completions).to(bandit_device)
            with torch.no_grad(): probs = q_bandit(feats)
            rewards = [float(r + bandit_alpha*(p.item()-0.5)) for r,p in zip(rewards, probs)]
            with torch.enable_grad():
                q_bandit_opt.zero_grad()
                probs2 = q_bandit(feats)
                baseline = float(np.mean([max(0.0, rr) for rr in rewards]))
                target = torch.tensor([1.0 if rr >= baseline else 0.0 for rr in rewards],
                                      dtype=torch.float32, device=probs2.device)
                F.binary_cross_entropy(probs2, target).backward(); q_bandit_opt.step()
        return rewards

    def mixed_reward(prompts=None, completions=None, task=None, **kw):
        out = []
        for i,t in enumerate(task):
            if t == "math":
                out.extend(reward_math(
                    prompts=[prompts[i]], completions=[completions[i]], final=[kw.get("final",[None])[i]]
                ))
            else:
                out.extend(reward_code(
                    prompts=[prompts[i]], completions=[completions[i]],
                    test_setup_code=[kw.get("test_setup_code", [""])[i]],
                    tests=[kw.get("tests", [[]])[i]],
                ))
        return out

    # Explicit generation kwargs to prevent warnings and improve completions
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
        logging_steps=5, logging_first_step=True,
        disable_tqdm=False, report_to="none",
        num_generations=group_size,
        generation_batch_size=group_size,
        temperature=rl_temperature, top_p=rl_top_p, top_k=rl_top_k,
        max_prompt_length=max_len, 
        max_completion_length=max_completion_length,  # Use the parameter
        generation_kwargs=generation_kwargs,
        gradient_checkpointing=False,  # RL: keep OFF
    )

    trainer = GRPOTrainer(
        model=sft_model, processing_class=tok, reward_funcs=mixed_reward,
        train_dataset=mix, args=args,
    )
    return trainer, tok

# ===================== Main =====================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", type=str, default="google/gemma-3-1b-it")
    ap.add_argument("--output_dir", type=str, default="./runs/gemma-3-1b-qrl")
    ap.add_argument("--sft_steps", type=int, default=300)
    ap.add_argument("--rl_steps", type=int, default=300)
    ap.add_argument("--skip_sft", action="store_true", help="Skip SFT phase and go straight to GRPO using --sft_ckpt")
    ap.add_argument("--sft_ckpt", type=str, default=None, help="Path to existing SFT output dir or specific checkpoint-* directory")
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--grad_acc", type=int, default=8)
    ap.add_argument("--group_size", type=int, default=4)
    ap.add_argument("--max_len", type=int, default=4096)
    ap.add_argument("--bf16", action="store_true"); ap.add_argument("--no_bf16", dest="bf16", action="store_false")
    ap.set_defaults(bf16=True)

    ap.add_argument("--enable_q_bandit", action="store_true", help="Enable Quantum Bandit for reward shaping.")
    ap.add_argument("--bandit_alpha", type=float, default=0.2)
    ap.add_argument("--enable_qk_prm", dest="enable_qk_prm", action="store_true", help="Enable Quantum Kernel PRM for reward shaping (default: enabled).")
    ap.add_argument("--disable_qk_prm", dest="enable_qk_prm", action="store_false", help="Disable Quantum Kernel PRM.")
    ap.set_defaults(enable_qk_prm=True)
    ap.add_argument("--prm_weight", type=float, default=0.3)

    # RL sampling
    ap.add_argument("--rl_top_p", type=float, default=0.9)
    ap.add_argument("--rl_top_k", type=int,   default=40)
    ap.add_argument("--rl_temperature", type=float, default=0.8)
    ap.add_argument("--max_completion_length", type=int, default=256, help="Max completion length for GRPO (shorter = faster, encourages concise answers)")

    # Eval sampling & batching
    ap.add_argument("--eval_sample", action="store_true")
    ap.add_argument("--eval_top_p", type=float, default=0.9)
    ap.add_argument("--eval_top_k", type=int,   default=40)
    ap.add_argument("--eval_temperature", type=float, default=0.7)
    ap.add_argument("--eval_batch_size", type=int, default=8)
    ap.add_argument("--eval_prompt_max_len", type=int, default=2048)
    ap.add_argument("--skip_baseline_eval", action="store_true")
    ap.add_argument("--eval_only", action="store_true", help="Only run evaluation on an existing GRPO adapter (requires --grpo_ckpt or --sft_ckpt)")
    ap.add_argument("--grpo_ckpt", type=str, default=None, help="Path to GRPO adapter checkpoint (overrides sft_ckpt if provided)")

    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    baseline = None
    if not args.skip_baseline_eval:
        tok0 = AutoTokenizer.from_pretrained(args.base_model, use_fast=True, trust_remote_code=True)
        if tok0.pad_token is None:
            tok0.pad_token = tok0.eos_token
        tok0.padding_side = "right"

        bnb0 = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        base = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb0,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager",
        )
        _force_bf16(base)
        base.config.use_cache = False

        gc0 = safe_clone_gen_cfg(getattr(base, "generation_config", None))
        gc0.do_sample = False
        gc0.top_p = None
        gc0.top_k = None
        gc0.temperature = None
        gc0.cache_implementation = "hybrid"
        base.generation_config = gc0

        baseline = eval_all(
            "baseline",
            base,
            tok0,
            args.output_dir,
            do_sample=args.eval_sample,
            top_p=args.eval_top_p if args.eval_sample else None,
            top_k=args.eval_top_k if args.eval_sample else None,
            temperature=args.eval_temperature if args.eval_sample else None,
            batch_size=args.eval_batch_size,
            eval_prompt_max_len=args.eval_prompt_max_len,
        )
        print("[Baseline]", json.dumps(baseline, indent=2))
    else:
        print("[Baseline] Skipped baseline evaluation (--skip_baseline_eval flag used)")

    if args.skip_sft:
        if not args.sft_ckpt:
            raise SystemExit("--skip_sft requires --sft_ckpt to point to an existing SFT directory or checkpoint-*/ path")
        sft_ckpt = args.sft_ckpt
        print(f"[Main] Skipping SFT. Using provided checkpoint path: {sft_ckpt}")
    else:
        sft_trainer, tok_sft = build_sft_trainer(
            base_model=args.base_model, output_dir=args.output_dir,
            sft_steps=args.sft_steps, grad_acc=args.grad_acc, bf16=args.bf16, max_len=args.max_len
        )
        sft_trainer.train()
        sft_ckpt = sft_trainer.args.output_dir

    if args.eval_only:
        # Load model (base + adapter) and run eval_all("trained")
        adapter_path = args.grpo_ckpt or args.sft_ckpt or sft_ckpt
        if not adapter_path:
            raise SystemExit("--eval_only requires --grpo_ckpt or --sft_ckpt to locate an adapter")
        print(f"[EvalOnly] Loading adapter from: {adapter_path}")
        tok_eval = AutoTokenizer.from_pretrained(args.base_model, use_fast=True, trust_remote_code=True)
        if tok_eval.pad_token is None: tok_eval.pad_token = tok_eval.eos_token
        tok_eval.padding_side = "right"  # switched internally to left during eval
        bnb_eval = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                                      bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
        base_eval = AutoModelForCausalLM.from_pretrained(
            args.base_model, torch_dtype=torch.bfloat16, quantization_config=bnb_eval,
            device_map="auto", trust_remote_code=True, attn_implementation="eager"
        )
        _force_bf16(base_eval)
        if os.path.isdir(adapter_path):
            # Accept either direct checkpoint-* or parent dir
            if not os.path.basename(adapter_path).startswith("checkpoint-"):
                cands = glob.glob(os.path.join(adapter_path, "checkpoint-*"))
                if cands:
                    adapter_path = max(cands, key=os.path.getctime)
        try:
            model_eval = PeftModel.from_pretrained(base_eval, adapter_path, is_trainable=False)
            print(f"[EvalOnly] Adapter loaded: {adapter_path}")
        except Exception as e:
            print(f"[EvalOnly] Failed to load adapter at {adapter_path}: {e}. Using base model only.")
            model_eval = base_eval
        metrics_trained = eval_all(
            "trained", model_eval, tok_eval, args.output_dir,
            do_sample=args.eval_sample,
            top_p=args.eval_top_p if args.eval_sample else None,
            top_k=args.eval_top_k if args.eval_sample else None,
            temperature=args.eval_temperature if args.eval_sample else None,
            batch_size=args.eval_batch_size,
            eval_prompt_max_len=args.eval_prompt_max_len
        )
        print("[EvalOnly Metrics]", json.dumps(metrics_trained, indent=2))
        return
    else:
        # GRPO (fixed trainables)
        grpo_trainer, tok_rl = build_grpo_trainer(
            base_model=args.base_model, sft_ckpt=sft_ckpt, output_dir=args.output_dir,
            rl_steps=args.rl_steps, per_device_train_batch_size=args.per_device_train_batch_size,
            grad_acc=max(1, args.grad_acc//2), group_size=args.group_size, bf16=args.bf16, max_len=args.max_len,
            enable_q_bandit=args.enable_q_bandit, bandit_alpha=args.bandit_alpha,
            enable_qk_prm=args.enable_qk_prm, prm_weight=args.prm_weight,
            rl_top_p=args.rl_top_p, rl_top_k=args.rl_top_k, rl_temperature=args.rl_temperature,
            max_completion_length=args.max_completion_length
        )
        grpo_trainer.train()

    # Post-train evals
    trained = eval_all(
        "trained", grpo_trainer.model, tok_rl, args.output_dir,
        do_sample=args.eval_sample,
        top_p=args.eval_top_p if args.eval_sample else None,
        top_k=args.eval_top_k if args.eval_sample else None,
        temperature=args.eval_temperature if args.eval_sample else None,
        batch_size=args.eval_batch_size,
        eval_prompt_max_len=args.eval_prompt_max_len
    )
    print("[Trained]", json.dumps(trained, indent=2))
    if baseline is not None:
        print_compare(baseline, trained)
    else:
        print("Baseline comparison skipped (baseline evaluation was not run)")

    if baseline is not None:
        print(f"Saved metrics to:\n  {os.path.join(args.output_dir,'metrics_baseline.json')}\n  {os.path.join(args.output_dir,'metrics_trained.json')}")
    else:
        print(f"Saved metrics to:\n  {os.path.join(args.output_dir,'metrics_trained.json')}")
    print(f"SFT ckpt:  {sft_ckpt}\nGRPO ckpt: {grpo_trainer.args.output_dir}")
    if args.skip_sft:
        print("[Info] SFT phase skipped; only GRPO adapter updates (if any) were applied.")

if __name__ == "__main__":
    main()
