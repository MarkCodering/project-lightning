# -*- coding: utf-8 -*-
"""
Baseline → SFT → GRPO pipeline with automatic before/after benchmarks
Base: google/gemma-3-270m

Benchmarks:
- Math: GSM8K (exact-match via '#### final' parsing)
- Code: MBPP (Muennighoff) + HumanEval (pass@1 via provided tests)

Outputs:
- JSON metrics in output_dir/metrics_baseline.json and metrics_trained.json
- Printed comparison table

Run:
  accelerate launch train_and_eval_gemma3_270m.py \
    --output_dir ./checkpoints/gemma3_270m_reasoner \
    --sft_steps 300 --rl_steps 300
"""

import os, re, json, argparse, tempfile, subprocess
from typing import List, Tuple

import torch
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig, GRPOTrainer, GRPOConfig

# ------------- tiny utils -------------
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
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", txt, re.S|re.I)
    if m: return _norm(m.group(1))
    tail = re.findall(r"[-+]?\d*\.?\d+(?:/\d+)?", txt)
    return _norm(tail[-1]) if tail else _norm(txt[-64:])

# ------------- prompts -------------
MATH_SYS = "You are a careful mathematician. Solve step by step in <think>...</think>, then put ONLY the final short answer inside <answer>...</answer>."
CODE_SYS = "You are a helpful coding assistant. Write correct, efficient Python code. Return ONLY code if tests will run."

def p_math(q: str) -> str:
    return f"{MATH_SYS}\n\n<task>\n{q.strip()}\n</task>"

def t_math(thought: str, final: str) -> str:
    return f"<think>\n{thought.strip()}\n</think>\n<answer>{final.strip()}</answer>"

def p_code(desc: str) -> str:
    return f"{CODE_SYS}\n\nProblem:\n{desc.strip()}\n\nWrite the solution."

# ------------- datasets -------------
def load_math_sft():
    gsm = load_dataset("openai/gsm8k", "main", split="train")
    gsm = gsm.map(lambda ex: {
        "source": "gsm8k",
        "prompt": p_math(ex["question"]),
        "target": t_math(ex["answer"], extract_gsm8k_final(ex["answer"])),
        "final": extract_gsm8k_final(ex["answer"])
    })
    math = load_dataset("EleutherAI/hendrycks_math", split="train")
    math = math.map(lambda ex: {
        "source": "math",
        "prompt": p_math(ex["problem"]),
        "target": t_math(ex["solution"], extract_math_final(ex["solution"])),
        "final": extract_math_final(ex["solution"])
    })
    return concatenate_datasets([gsm, math])

def load_code_sets():
    mbpp = load_dataset("Muennighoff/mbpp", split="test")
    he = load_dataset("openai/openai_humaneval", split="test")
    def map_mbpp(ex):
        return {
            "source": "mbpp",
            "desc": ex["text"],
            "test_setup_code": ex.get("test_setup_code", ""),
            "tests": ex.get("test_list", []),
        }
    mbpp = mbpp.map(map_mbpp)
    def map_he(ex):
        return {
            "source": "humaneval",
            "desc": ex["prompt"],
            "test_setup_code": "",
            "tests": [ex["test"]],
        }
    he = he.map(map_he)
    return mbpp, he

# ------------- sandbox (unsafe for prod) -------------
def run_code_and_tests(code: str, setup: str, tests: List[str], timeout_s=4) -> Tuple[bool, str]:
    snippet = "\n".join([setup or "", code, "\n".join(tests)])
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(snippet)
        path = f.name
    try:
        proc = subprocess.run(["python", path], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True, timeout=timeout_s)
        return proc.returncode == 0, proc.stderr
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    finally:
        try: os.remove(path)
        except: pass

# ------------- evals -------------
@torch.inference_mode()
def eval_gsm8k(model, tok, n=100, max_new_tokens=256):
    ds = load_dataset("openai/gsm8k", "main", split="test").select(range(n))
    ok = 0
    for ex in ds:
        prompt = p_math(ex["question"])
        ids = tok(prompt, return_tensors="pt").to(model.device)
        out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False)
        txt = tok.decode(out[0], skip_special_tokens=True)[len(prompt):]
        pred = parse_final_from_completion(txt)
        if equal_numeric(pred, extract_gsm8k_final(ex["answer"])): ok += 1
    return ok / n

@torch.inference_mode()
def eval_mbpp(model, tok, n=50, max_new_tokens=200):
    mbpp = load_dataset("Muennighoff/mbpp", split="test").select(range(n))
    passed = 0
    for ex in mbpp:
        prompt = p_code(ex["text"])
        ids = tok(prompt, return_tensors="pt").to(model.device)
        out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False)
        txt = tok.decode(out[0], skip_special_tokens=True)[len(prompt):]
        m = re.search(r"```(?:python)?\s*(.*?)```", txt, re.S|re.I)
        code = m.group(1) if m else txt
        ok, _ = run_code_and_tests(code, ex.get("test_setup_code",""), ex.get("test_list",[]))
        if ok: passed += 1
    return passed / n

@torch.inference_mode()
def eval_humaneval(model, tok, n=50, max_new_tokens=200):
    he = load_dataset("openai/openai_humaneval", split="test").select(range(n))
    passed = 0
    for ex in he:
        prompt = p_code(ex["prompt"])
        ids = tok(prompt, return_tensors="pt").to(model.device)
        out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False)
        txt = tok.decode(out[0], skip_special_tokens=True)[len(prompt):]
        m = re.search(r"```(?:python)?\s*(.*?)```", txt, re.S|re.I)
        code = m.group(1) if m else txt
        ok, _ = run_code_and_tests(code, "", [ex["test"]])
        if ok: passed += 1
    return passed / n

def eval_all(label, model, tok, outdir):
    metrics = {
        "label": label,
        "gsm8k_acc": eval_gsm8k(model, tok, n=100),
        "mbpp_pass1": eval_mbpp(model, tok, n=50),
        "humaneval_pass1": eval_humaneval(model, tok, n=30),
    }
    path = os.path.join(outdir, f"metrics_{label}.json")
    with open(path, "w") as f: json.dump(metrics, f, indent=2)
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

# ------------- SFT & GRPO builders -------------
def build_sft_trainer(base_model, output_dir, sft_steps=300, lr=2e-5,
                      per_device_train_batch_size=1, grad_acc=8, bf16=True, max_len=4096):
    tok = AutoTokenizer.from_pretrained(base_model, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, quantization_config=bnb,
        device_map="auto", trust_remote_code=True
    )

    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
    )

    args = SFTConfig(
        output_dir=os.path.join(output_dir, "sft"),
        max_steps=sft_steps,
        learning_rate=lr,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=grad_acc,
        bf16=bf16,
        logging_steps=10, save_steps=200,
        max_seq_length=max_len, packing=False,
        gradient_checkpointing=True, lr_scheduler_type="cosine", warmup_ratio=0.03,
    )

    math_sft = load_math_sft()
    def formatting(batch):
        return [f'{p}\n\n{t}' for p,t in zip(batch["prompt"], batch["target"])]

    trainer = SFTTrainer(
        model=model, tokenizer=tok, args=args, train_dataset=math_sft,
        formatting_func=formatting, peft_config=lora,
    )
    return trainer, tok

def reward_math(prompts=None, completions=None, final=None, **kwargs):
    rewards = []
    for comp, gt in zip(completions, final):
        pred = parse_final_from_completion(comp)
        rewards.append(1.0 if equal_numeric(pred, gt) else 0.0)
    return rewards

def run_code_reward(completions, setups, tests_lists):
    scores = []
    for comp, setup, tlist in zip(completions, setups, tests_lists):
        m = re.search(r"```(?:python)?\s*(.*?)```", comp, re.S|re.I)
        code = m.group(1) if m else comp
        ok, _ = run_code_and_tests(code, setup, tlist)
        scores.append(1.0 if ok else 0.0)
    return scores

def build_grpo_trainer(base_model, sft_ckpt, output_dir, rl_steps=300,
                       per_device_train_batch_size=1, grad_acc=4, bf16=True,
                       group_size=4, max_len=4096):
    tok = AutoTokenizer.from_pretrained(base_model, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    # RL datasets
    math_train = load_math_sft().remove_columns(["target"]).rename_column("final","final")
    math_train = math_train.map(lambda ex: {"task":"math"})

    mbpp, he = load_code_sets()
    code_train = concatenate_datasets([mbpp, he]).map(lambda ex: {"prompt": p_code(ex["desc"]), "task":"code"})

    mix = concatenate_datasets([math_train, code_train]).shuffle(seed=42)

    def mixed_reward(prompts=None, completions=None, task=None, **kw):
        out = []
        for i,t in enumerate(task):
            if t == "math":
                out.append(reward_math(
                    prompts=[prompts[i]], completions=[completions[i]],
                    final=[kw.get("final",[None])[i]]
                )[0])
            else:
                out.append(run_code_reward(
                    [completions[i]],
                    [kw.get("test_setup_code",[ ""])[i]],
                    [kw.get("tests", [[]])[i]]
                )[0])
        return out

    args = GRPOConfig(
        output_dir=os.path.join(output_dir, "grpo"),
        max_steps=rl_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=grad_acc,
        bf16=bf16, logging_steps=10, save_steps=200,
        # generations
        num_generations=group_size, temperature=0.8, top_p=0.9,
        max_prompt_length=max_len, max_completion_length=512,
        # regularization
        kl_coeff=0.05,
        gradient_checkpointing=True,
    )

    trainer = GRPOTrainer(
        model=sft_ckpt, tokenizer=tok, reward_funcs=mixed_reward,
        train_dataset=mix, args=args,
    )
    return trainer, tok

# ------------- main -------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", type=str, default="google/gemma-3-270m")
    ap.add_argument("--output_dir", type=str, default="./checkpoints/gemma3_270m_reasoner")
    ap.add_argument("--sft_steps", type=int, default=300)
    ap.add_argument("--rl_steps", type=int, default=300)
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--grad_acc", type=int, default=8)
    ap.add_argument("--group_size", type=int, default=4)
    ap.add_argument("--max_len", type=int, default=4096)
    ap.add_argument("--bf16", action="store_true"); ap.add_argument("--no_bf16", dest="bf16", action="store_false")
    ap.set_defaults(bf16=True)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ----- Load base for baseline eval -----
    tok0 = AutoTokenizer.from_pretrained(args.base_model, use_fast=True, trust_remote_code=True)
    if tok0.pad_token is None: tok0.pad_token = tok0.eos_token

    bnb0 = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                              bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, quantization_config=bnb0,
        device_map="auto", trust_remote_code=True
    )

    # ----- Baseline evals -----
    baseline = eval_all("baseline", base, tok0, args.output_dir)
    print("[Baseline metrics]", json.dumps(baseline, indent=2))

    # ----- SFT -----
    sft_trainer, tok_sft = build_sft_trainer(
        base_model=args.base_model, output_dir=args.output_dir,
        sft_steps=args.sft_steps, grad_acc=args.grad_acc, bf16=args.bf16, max_len=args.max_len
    )
    sft_trainer.train()
    sft_ckpt = sft_trainer.args.output_dir   # path to ./sft

    # ----- GRPO -----
    grpo_trainer, tok_rl = build_grpo_trainer(
        base_model=args.base_model, sft_ckpt=sft_ckpt, output_dir=args.output_dir,
        rl_steps=args.rl_steps, per_device_train_batch_size=args.per_device_train_batch_size,
        grad_acc=max(1, args.grad_acc//2), group_size=args.group_size, bf16=args.bf16, max_len=args.max_len
    )
    grpo_trainer.train()

    # Trained model handle
    trained_model = grpo_trainer.model
    trained_tok = tok_rl

    # ----- Post-train evals -----
    trained = eval_all("trained", trained_model, trained_tok, args.output_dir)
    print("[Trained metrics]", json.dumps(trained, indent=2))

    # ----- Comparison -----
    print_compare(baseline, trained)
    print(f"Saved metrics to:\n  {os.path.join(args.output_dir,'metrics_baseline.json')}\n  {os.path.join(args.output_dir,'metrics_trained.json')}")
    print(f"SFT ckpt:  {sft_ckpt}\nGRPO ckpt: {grpo_trainer.args.output_dir}")

if __name__ == "__main__":
    main()

