#!/usr/bin/env python3
"""
Package trained adapter weights with pretrained model for inference.

This script merges Stage 1 trained adapter weights with pretrained
encoder and LoRA weights to create an inference-ready model checkpoint.
"""

import argparse
import hashlib
import torch
from pathlib import Path


def compute_checksum(state_dict, keys):
    """
    Compute MD5 checksum for a subset of parameters.

    Args:
        state_dict: Model state dictionary
        keys: List of keys to include in checksum

    Returns:
        MD5 checksum string
    """
    hasher = hashlib.md5()
    for key in sorted(keys):
        if key in state_dict:
            tensor_bytes = state_dict[key].detach().cpu().numpy().tobytes()
            hasher.update(tensor_bytes)
    return hasher.hexdigest()


def validate_adapter_shapes(pretrained_dict, trained_dict, adapter_keys):
    """
    Verify that adapter tensor shapes match between checkpoints.

    Args:
        pretrained_dict: Pretrained model state dict
        trained_dict: Trained adapter state dict
        adapter_keys: List of adapter parameter keys

    Raises:
        ValueError: If shapes don't match
    """
    mismatches = []

    for key in adapter_keys:
        if key not in pretrained_dict:
            raise KeyError(f"Adapter key '{key}' not found in pretrained model")
        if key not in trained_dict:
            raise KeyError(f"Adapter key '{key}' not found in trained checkpoint")

        pretrained_shape = pretrained_dict[key].shape
        trained_shape = trained_dict[key].shape

        if pretrained_shape != trained_shape:
            mismatches.append(
                f"  {key}: pretrained={pretrained_shape}, trained={trained_shape}"
            )

    if mismatches:
        raise ValueError(
            "Adapter shape mismatches detected:\n" + "\n".join(mismatches) +
            "\n\nThis indicates the adapter architecture changed. "
            "Check encoder_downsample_rate or adapter structure."
        )


def count_parameters(state_dict, key_pattern=None):
    """Count total parameters, optionally filtered by key pattern."""
    count = 0
    for key, tensor in state_dict.items():
        if key_pattern is None or key_pattern in key:
            count += tensor.numel()
    return count


def merge_adapter_weights(pretrained_path, trained_path, output_path):
    """
    Main merging logic: combine trained adapter with pretrained model.

    Args:
        pretrained_path: Path to pretrained model.pth.tar
        trained_path: Path to training checkpoint epoch-N.pt
        output_path: Path to save packaged model
    """
    print("=" * 60)
    print("FireRedASR Model Packaging")
    print("=" * 60)

    # 1. Load pretrained model
    print(f"\nLoading pretrained model from: {pretrained_path}")
    pretrained = torch.load(pretrained_path, map_location='cpu', weights_only=False)

    if 'model_state_dict' not in pretrained or 'args' not in pretrained:
        raise ValueError(
            "Pretrained checkpoint format incorrect. "
            "Expected keys: ['model_state_dict', 'args']"
        )

    pretrained_state = pretrained['model_state_dict']
    pretrained_args = pretrained['args']

    # Count parameters by component
    total_params = count_parameters(pretrained_state)
    encoder_params = count_parameters(pretrained_state, 'encoder.')
    lora_params = count_parameters(pretrained_state, 'lora')
    adapter_params = count_parameters(pretrained_state, 'encoder_projector.')

    print(f"  Total parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"  Encoder parameters: {encoder_params:,} ({encoder_params/1e6:.1f}M)")
    print(f"  Adapter parameters: {adapter_params:,} ({adapter_params/1e6:.1f}M)")

    lora_keys = [k for k in pretrained_state.keys() if 'lora' in k.lower()]
    print(f"  LoRA parameters: {lora_params:,} ({lora_params/1e6:.1f}M, {len(lora_keys)} keys)")

    # 2. Load trained adapter
    print(f"\nLoading trained adapter from: {trained_path}")
    trained_ckpt = torch.load(trained_path, map_location='cpu', weights_only=False)

    if 'model' not in trained_ckpt:
        raise ValueError(
            "Training checkpoint format incorrect. Expected key: 'model'"
        )

    trained_state = trained_ckpt['model']
    trained_adapter_keys = [k for k in trained_state.keys() if 'encoder_projector' in k]

    print(f"  Found {len(trained_adapter_keys)} adapter parameters:")
    for key in sorted(trained_adapter_keys):
        shape = trained_state[key].shape
        params = trained_state[key].numel()
        print(f"    {key}: {shape} ({params:,} params)")

    # 3. Define adapter keys to replace
    adapter_keys = [
        'encoder_projector.linear1.weight',
        'encoder_projector.linear1.bias',
        'encoder_projector.linear2.weight',
        'encoder_projector.linear2.bias',
    ]

    # 4. Validate shapes
    print("\nValidating adapter shapes...")
    validate_adapter_shapes(pretrained_state, trained_state, adapter_keys)
    print("  ✓ All shapes match")

    # 5. Compute checksums before replacement
    print("\nComputing checksums for verification...")
    encoder_keys = [k for k in pretrained_state.keys() if k.startswith('encoder.')]
    lora_keys = [k for k in pretrained_state.keys() if 'lora' in k.lower()]

    encoder_checksum_before = compute_checksum(pretrained_state, encoder_keys)
    lora_checksum_before = compute_checksum(pretrained_state, lora_keys)
    adapter_checksum_before = compute_checksum(pretrained_state, adapter_keys)

    print(f"  Encoder checksum: {encoder_checksum_before[:8]}...")
    print(f"  LoRA checksum: {lora_checksum_before[:8]}...")
    print(f"  Adapter checksum (old): {adapter_checksum_before[:8]}...")

    # 6. Replace adapter weights
    print("\nReplacing adapter weights...")
    for key in adapter_keys:
        old_shape = pretrained_state[key].shape
        new_shape = trained_state[key].shape
        pretrained_state[key] = trained_state[key].clone()
        print(f"  ✓ {key}: {old_shape} -> {new_shape}")

    # 7. Verify replacement
    print("\nVerifying replacement...")
    encoder_checksum_after = compute_checksum(pretrained_state, encoder_keys)
    lora_checksum_after = compute_checksum(pretrained_state, lora_keys)
    adapter_checksum_after = compute_checksum(pretrained_state, adapter_keys)

    if encoder_checksum_before != encoder_checksum_after:
        raise RuntimeError("ERROR: Encoder weights changed unexpectedly!")
    print(f"  ✓ Encoder unchanged ({encoder_checksum_after[:8]}...)")

    if lora_checksum_before != lora_checksum_after:
        raise RuntimeError("ERROR: LoRA weights changed unexpectedly!")
    print(f"  ✓ LoRA unchanged ({lora_checksum_after[:8]}...)")

    if adapter_checksum_before == adapter_checksum_after:
        raise RuntimeError("ERROR: Adapter weights did not change!")
    print(f"  ✓ Adapter replaced ({adapter_checksum_after[:8]}...)")

    # 8. Save packaged model
    print(f"\nSaving packaged model to: {output_path}")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    packaged_model = {
        'model_state_dict': pretrained_state,
        'args': pretrained_args,
    }

    torch.save(packaged_model, output_path)

    file_size_gb = output_path.stat().st_size / (1024**3)
    print(f"  ✓ Saved successfully")
    print(f"  Size: {file_size_gb:.2f} GB")
    print(f"  Format: {{'model_state_dict': ..., 'args': ...}}")

    print("\n" + "=" * 60)
    print("Packaging complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Copy required files to model directory:")
    print(f"       cp <pretrained_dir>/cmvn.ark {output_path.parent}/")
    print(f"       ln -sf <pretrained_dir>/asr_encoder.pth.tar {output_path.parent}/")
    print(f"       ln -sf <pretrained_dir>/Qwen2-7B-Instruct {output_path.parent}/")
    print("  2. Run inference validation:")
    print(f"       python scripts/test_inference.py --model-dir {output_path.parent}")


def main():
    parser = argparse.ArgumentParser(
        description='Package trained adapter weights with pretrained model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python scripts/package_model.py \\
    --pretrained-model /path/to/FireRedASR-LLM-L/model.pth.tar \\
    --trained-adapter exp/test_stage1/epoch-3.pt \\
    --output-model exp/test_stage1/model.pth.tar
        """
    )

    parser.add_argument(
        '--pretrained-model',
        type=str,
        required=True,
        help='Path to pretrained model.pth.tar (contains encoder + adapter + LoRA)'
    )

    parser.add_argument(
        '--trained-adapter',
        type=str,
        required=True,
        help='Path to training checkpoint (epoch-N.pt) with trained adapter'
    )

    parser.add_argument(
        '--output-model',
        type=str,
        required=True,
        help='Path to save packaged model.pth.tar'
    )

    args = parser.parse_args()

    # Validate input files exist
    if not Path(args.pretrained_model).exists():
        raise FileNotFoundError(f"Pretrained model not found: {args.pretrained_model}")

    if not Path(args.trained_adapter).exists():
        raise FileNotFoundError(f"Training checkpoint not found: {args.trained_adapter}")

    # Run merging
    merge_adapter_weights(
        args.pretrained_model,
        args.trained_adapter,
        args.output_model
    )


if __name__ == '__main__':
    main()
