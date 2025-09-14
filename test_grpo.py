#!/usr/bin/env python3
"""Quick test to check GRPOTrainer and GRPOConfig parameters."""

try:
    from trl import GRPOTrainer, GRPOConfig
    import inspect
    
    print("GRPOConfig signature:")
    print(inspect.signature(GRPOConfig))
    print("\nGRPOTrainer signature:")
    print(inspect.signature(GRPOTrainer.__init__))
    
    # Test creating GRPOConfig
    try:
        config = GRPOConfig(
            output_dir="./test",
            max_steps=1,
            per_device_train_batch_size=1,
        )
        print("\nGRPOConfig creation: SUCCESS")
    except Exception as e:
        print(f"\nGRPOConfig creation ERROR: {e}")
        
except ImportError as e:
    print(f"Import error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
