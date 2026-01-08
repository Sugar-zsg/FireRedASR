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
    修改版：支持合并 Adapter 和 LoRA 权重
    """
    print("=" * 60)
    print("FireRedASR Model Packaging (With LoRA Support)")
    print("=" * 60)

    # 1. Load pretrained model
    print(f"\nLoading pretrained model from: {pretrained_path}")
    pretrained = torch.load(pretrained_path, map_location='cpu', weights_only=False)
    pretrained_state = pretrained['model_state_dict']
    pretrained_args = pretrained['args']

    # 2. Load trained weights (兼容 .bin 和 .pt)
    print(f"\nLoading trained weights from: {trained_path}")
    trained_ckpt = torch.load(trained_path, map_location='cpu', weights_only=False)

    # 兼容处理：获取 state_dict
    if isinstance(trained_ckpt, dict) and 'model' in trained_ckpt:
        raw_trained_state = trained_ckpt['model']
    elif isinstance(trained_ckpt, dict) and 'state_dict' in trained_ckpt:
        raw_trained_state = trained_ckpt['state_dict']
    else:
        raw_trained_state = trained_ckpt  # 假设本身就是 state_dict (如 pytorch_model.bin)

    # --- [关键修改 A]：清洗 Key 前缀 ---
    # LoRA 训练常见的冗余前缀，需要去除才能匹配到底模
    trained_state = {}
    prefixes_to_remove = ["base_model.model.", "module.", "model."]

    print(f"  Processing {len(raw_trained_state)} keys from trained file...")
    for k, v in raw_trained_state.items():
        new_k = k
        for p in prefixes_to_remove:
            if new_k.startswith(p):
                new_k = new_k.replace(p, "", 1)
        trained_state[new_k] = v

    # --- [关键修改 B]：动态构建要替换的 Key 列表 ---
    # 我们只替换 Adapter 和 LoRA，不替换 Encoder 主干（防止意外覆盖）
    keys_to_replace = []

    for key in trained_state.keys():
        # 条件1: 必须在底模中存在
        if key not in pretrained_state:
            continue

        # 条件2: 必须是 Adapter 或 LoRA 相关的参数
        # 注意：这里假设 LoRA 参数名包含 'lora'
        if 'encoder_projector' in key or 'lora' in key.lower():
            keys_to_replace.append(key)

    if not keys_to_replace:
        raise ValueError("没有找到匹配的 Adapter 或 LoRA 权重！请检查 Key 的名称前缀。")

    print(f"  Found {len(keys_to_replace)} keys to merge (Adapter + LoRA).")

    # 4. Validate shapes (只检查我们要替换的那些)
    print("\nValidating shapes...")
    validate_adapter_shapes(pretrained_state, trained_state, keys_to_replace)
    print("  ✓ All shapes match")

    # 5. Compute checksums (计算替换前的哈希值)
    print("\nComputing checksums before merge...")

    # 获取底模中所有的 LoRA 和 Adapter key 用于对比
    all_lora_keys = [k for k in pretrained_state.keys() if 'lora' in k.lower()]
    all_adapter_keys = [k for k in pretrained_state.keys() if 'encoder_projector' in k]

    lora_checksum_before = compute_checksum(pretrained_state, all_lora_keys)
    adapter_checksum_before = compute_checksum(pretrained_state, all_adapter_keys)

    # 6. Replace weights (核心替换步骤)
    print("\nReplacing weights...")
    for key in keys_to_replace:
        pretrained_state[key] = trained_state[key].clone()
        # 打印少量日志，避免刷屏
        if 'linear1.weight' in key or keys_to_replace.index(key) < 3:
            print(f"  ✓ Updated: {key}")

    print(f"  ... and {len(keys_to_replace) - 3} more keys.")

    # 7. Verify replacement (校验)
    print("\nVerifying replacement...")
    lora_checksum_after = compute_checksum(pretrained_state, all_lora_keys)
    adapter_checksum_after = compute_checksum(pretrained_state, all_adapter_keys)

    # --- [关键修改 C]：修改校验逻辑 ---

    # 检查 Adapter 是否更新
    if adapter_checksum_before == adapter_checksum_after:
        print("  ⚠ Warning: Adapter weights did NOT change. (Did you freeze them?)")
    else:
        print(f"  ✓ Adapter updated successfully.")

    # 检查 LoRA 是否更新 (原脚本这里如果变了会报错，现在改为希望它变)
    if lora_checksum_before == lora_checksum_after:
        print("  ℹ LoRA weights did NOT change (Validation: only Adapter trained?)")
    else:
        print(f"  ✓ LoRA updated successfully ({lora_checksum_after[:8]}...)")

    # 8. Save
    print(f"\nSaving packaged model to: {output_path}")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        'model_state_dict': pretrained_state,
        'args': pretrained_args,
    }, output_path)

    print(f"  ✓ Saved successfully. Size: {output_path.stat().st_size / (1024 ** 3):.2f} GB")


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
