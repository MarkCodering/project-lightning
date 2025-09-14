# project-lightning

Reasoning SFT → GRPO training for Gemma 3 family with built-in benchmarks (GSM8K, MBPP, HumanEval) and optional quantum reward shaping.

## Datasets

- Math SFT: GSM8K + Hendrycks MATH (selected configs). Always included.
- Code SFT (optional):
	- MBPP: tries multiple variants; included by default when code solutions are available.
	- HumanEval: optional; includes canonical solutions if present.
		- Important: Using HumanEval for training contaminates the benchmark. If you plan to report HumanEval, keep it disabled during SFT.
- RL (GRPO):
	- Math: GSM8K + MATH transformed to prompts with numeric exact-match rewards.
	- Code: MBPP and HumanEval tasks with unit-test rewards (on-device Python sandbox). Both enabled by default; disable via flags below.

## Key script

- `train_quantum_reasoner.py` — end-to-end SFT and GRPO with dataset toggles and fast batched eval.

## Usage (examples)

Train with math only for SFT; include MBPP and HumanEval in RL:

```bash
python train_quantum_reasoner.py \
	--base_model google/gemma-3-1b-it \
	--output_dir ./runs/gemma-3-1b-qrl \
	--sft_steps 300 --rl_steps 300 \
	--no_sft_humaneval --sft_include_mbpp \
	--rl_include_mbpp --rl_include_humaneval
```

Avoid HumanEval entirely (no contamination in either phase):

```bash
python train_quantum_reasoner.py \
	--no_sft_humaneval --no_rl_humaneval
```

Limit dataset sizes to speed up experiments:

```bash
python train_quantum_reasoner.py \
	--sft_mbpp_limit 100 \
	--rl_mbpp_limit 200 --rl_humaneval_limit 30
```

## Notes

- HumanEval in SFT is disabled by default; enable with `--sft_include_humaneval` only if you do not intend to report HumanEval scores as unbiased.
- The Python sandbox used for code rewards is a local subprocess and not secure. Use only in controlled environments.
