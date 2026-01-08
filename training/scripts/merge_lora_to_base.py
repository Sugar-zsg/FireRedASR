#!/usr/bin/env python3
"""
Merge LoRA weights into base LLM model offline.

This script loads a base model, applies LoRA weights from a checkpoint,
merges them into the base model, and saves the result as a new base model.

Usage:
    python training/scripts/merge_lora_to_base.py \
        --base-model pretrained_models/Qwen2-7B-Instruct \
        --lora-checkpoint pretrained_models/FireRedASR-LLM-L/model.pth.tar \
        --output-dir pretrained_models/Qwen2-7B-Instruct-FireRedASR-Merged \
        --validate
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Merge LoRA weights into base model")
    parser.add_argument(
        "--base-model",
        type=str,
        required=True,
        help="Path to base model (e.g., Qwen2-7B-Instruct)",
    )
    parser.add_argument(
        "--lora-checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint containing LoRA weights (model.pth.tar)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save merged model",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate merge by comparing outputs",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for merge (cuda/cpu)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "float32", "bfloat16"],
        help="Data type for model loading",
    )
    return parser.parse_args()


def load_checkpoint(checkpoint_path):
    """Load checkpoint and extract LoRA weights."""
    logging.info(f"Loading checkpoint from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Extract LoRA weights
    lora_keys = {k: v for k, v in state_dict.items() if "lora" in k.lower()}

    logging.info(f"Found {len(lora_keys)} LoRA weight keys")

    if not lora_keys:
        logging.error("No LoRA weights found in checkpoint!")
        logging.info("Checkpoint keys sample:")
        for i, k in enumerate(list(state_dict.keys())[:10]):
            logging.info(f"  {k}")
        sys.exit(1)

    return lora_keys


def create_lora_model_from_checkpoint(base_model, lora_weights):
    """Create a PEFT model and load LoRA weights from checkpoint."""
    logging.info("Creating LoRA model from checkpoint weights")

    # LoRA configuration matching FireRedASR-LLM
    lora_config = LoraConfig(
        r=64,
        lora_alpha=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "up_proj", "gate_proj", "down_proj",
        ],
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )

    # Wrap base model with LoRA
    model = get_peft_model(base_model, lora_config)

    # Load LoRA weights from checkpoint
    # Need to handle key name differences between checkpoint and PEFT model
    # Checkpoint keys: llm.base_model.model.model.layers.X...
    # PEFT keys: base_model.model.layers.X...

    model_state = model.state_dict()
    loaded_count = 0
    missing_keys = []

    for ckpt_key, ckpt_value in lora_weights.items():
        # Strip 'llm.' prefix if present
        if ckpt_key.startswith('llm.'):
            model_key = ckpt_key[4:]  # Remove 'llm.'
        else:
            model_key = ckpt_key

        if model_key in model_state:
            model_state[model_key] = ckpt_value
            loaded_count += 1
        else:
            missing_keys.append(model_key)

    model.load_state_dict(model_state, strict=False)

    logging.info(f"Loaded {loaded_count}/{len(lora_weights)} LoRA weights")
    if missing_keys:
        logging.warning(f"Missing {len(missing_keys)} keys (sample):")
        for k in missing_keys[:5]:
            logging.warning(f"  {k}")

    return model


def merge_lora(peft_model):
    """Merge LoRA weights into base model."""
    logging.info("Merging LoRA weights into base model...")

    # Use PEFT's built-in merge function
    merged_model = peft_model.merge_and_unload()

    logging.info("✅ LoRA weights merged successfully")

    return merged_model


def validate_merge(base_model, lora_model, merged_model, tokenizer):
    """Validate that merged model produces same outputs as LoRA model."""
    logging.info("Validating merge by comparing outputs...")

    # Test prompt
    test_text = "今天天气怎么样?"
    inputs = tokenizer(test_text, return_tensors="pt")

    device = next(merged_model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate with merged model
    with torch.no_grad():
        merged_output = merged_model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
        )
        merged_text = tokenizer.decode(merged_output[0], skip_special_tokens=True)

    # Generate with LoRA model
    lora_model = lora_model.to(device)
    with torch.no_grad():
        lora_output = lora_model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
        )
        lora_text = tokenizer.decode(lora_output[0], skip_special_tokens=True)

    logging.info(f"Merged model output: {merged_text}")
    logging.info(f"LoRA model output: {lora_text}")

    if merged_text == lora_text:
        logging.info("✅ Validation passed: outputs match!")
        return True
    else:
        logging.warning("⚠️ Validation failed: outputs differ")
        logging.warning("This may be due to numerical precision differences")
        return False


def get_dtype(dtype_str):
    """Convert dtype string to torch dtype."""
    if dtype_str == "float16":
        return torch.float16
    elif dtype_str == "float32":
        return torch.float32
    elif dtype_str == "bfloat16":
        return torch.bfloat16
    else:
        raise ValueError(f"Unknown dtype: {dtype_str}")


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dtype = get_dtype(args.dtype)

    # Load base model
    logging.info(f"Loading base model from {args.base_model}")
    logging.info(f"Using dtype: {dtype}, device: {args.device}")

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map=args.device if args.device == "cpu" else "auto",
        trust_remote_code=True,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
    )

    # Load LoRA weights from checkpoint
    lora_weights = load_checkpoint(args.lora_checkpoint)

    # Create PEFT model with LoRA weights
    logging.info("Creating PEFT model with LoRA")
    lora_model = create_lora_model_from_checkpoint(base_model, lora_weights)

    # Validate before merge (optional)
    if args.validate:
        logging.info("Testing LoRA model before merge...")
        test_text = "你好"
        inputs = tokenizer(test_text, return_tensors="pt")
        device = next(lora_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            output = lora_model.generate(**inputs, max_new_tokens=10)
            generated = tokenizer.decode(output[0], skip_special_tokens=True)
            logging.info(f"LoRA model test output: {generated}")

    # Merge LoRA into base model
    merged_model = merge_lora(lora_model)

    # Validate merge (if requested)
    if args.validate:
        validate_merge(base_model, lora_model, merged_model, tokenizer)

    # Save merged model
    logging.info(f"Saving merged model to {output_dir}")
    merged_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Print statistics
    logging.info("\n" + "="*60)
    logging.info("MERGE COMPLETE")
    logging.info("="*60)
    logging.info(f"Base model: {args.base_model}")
    logging.info(f"LoRA checkpoint: {args.lora_checkpoint}")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"LoRA weights merged: {len(lora_weights)}")

    total_params = sum(p.numel() for p in merged_model.parameters())
    logging.info(f"Total parameters: {total_params:,}")

    logging.info("\n" + "Next steps:")
    logging.info("1. Use the merged model as --llm-dir in training")
    logging.info("2. Train fresh LoRA on top of merged model")
    logging.info(f"\nExample:")
    logging.info(f"  --llm-dir {output_dir}")
    logging.info("="*60)


if __name__ == "__main__":
    main()
