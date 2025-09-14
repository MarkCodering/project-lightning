#!/usr/bin/env python3
"""Quick test to check GRPOConfig with our parameters."""

from trl import GRPOConfig

try:
    config = GRPOConfig(
        output_dir="./test",
        max_steps=1,
        per_device_train_batch_size=1,
        num_generations=4,
        generation_batch_size=4,  # Must be divisible by num_generations
        temperature=0.8, 
        top_p=0.9, 
        top_k=40,
        max_prompt_length=512, 
        max_completion_length=128,
    )
    print("GRPOConfig creation: SUCCESS")
    print(f"num_generations: {config.num_generations}")
    print(f"generation_batch_size: {config.generation_batch_size}")
    
except Exception as e:
    print(f"GRPOConfig creation ERROR: {e}")
