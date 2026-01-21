#!/usr/bin/env python3
"""
灵活的打包脚本：支持不同的LoRA rank配置

使用场景：
1. Stage 1: 只打包Adapter（无LoRA）
2. Stage 2: 打包Adapter + LoRA（支持不同rank）
"""

import argparse
import hashlib
import torch
from pathlib import Path


def validate_adapter_shapes(pretrained_dict, trained_dict, adapter_keys):
    """验证adapter形状匹配（不验证LoRA）"""
    mismatches = []

    for key in adapter_keys:
        if key not in pretrained_dict:
            print(f"  ⚠ Warning: Key '{key}' not in pretrained model (will be added)")
            continue
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
            "Adapter shape mismatches:\n" + "\n".join(mismatches)
        )


def merge_weights(pretrained_path, trained_path, output_path, skip_lora_validation=False):
    """
    灵活的权重合并

    Args:
        skip_lora_validation: 如果为True，跳过LoRA形状验证（允许不同rank）
    """
    print("=" * 60)
    print("FireRedASR 模型打包（支持灵活LoRA配置）")
    print("=" * 60)

    # 1. 加载预训练模型
    print(f"\n[1/6] 加载预训练模型: {pretrained_path}")
    pretrained = torch.load(pretrained_path, map_location='cpu', weights_only=False)
    pretrained_state = pretrained['model_state_dict']
    pretrained_args = pretrained['args']
    print(f"  ✓ 预训练模型参数数量: {len(pretrained_state)}")

    # 2. 加载训练权重
    print(f"\n[2/6] 加载训练权重: {trained_path}")
    trained_ckpt = torch.load(trained_path, map_location='cpu', weights_only=False)

    # 兼容不同的checkpoint格式
    if isinstance(trained_ckpt, dict) and 'model' in trained_ckpt:
        raw_trained_state = trained_ckpt['model']
    elif isinstance(trained_ckpt, dict) and 'state_dict' in trained_ckpt:
        raw_trained_state = trained_ckpt['state_dict']
    else:
        raw_trained_state = trained_ckpt

    # 3. 清理key前缀
    print(f"\n[3/6] 清理key前缀...")
    trained_state = {}
    prefixes_to_remove = ["base_model.model.", "module.", "model."]

    for k, v in raw_trained_state.items():
        new_k = k
        for p in prefixes_to_remove:
            if new_k.startswith(p):
                new_k = new_k.replace(p, "", 1)
        trained_state[new_k] = v
    print(f"  ✓ 处理后参数数量: {len(trained_state)}")

    # 4. 分离Adapter和LoRA权重
    print(f"\n[4/6] 分析权重...")
    adapter_keys = []
    lora_keys = []

    for key in trained_state.keys():
        if 'encoder_projector' in key:
            adapter_keys.append(key)
        elif 'lora' in key.lower():
            lora_keys.append(key)

    print(f"  - Adapter权重: {len(adapter_keys)} 个")
    print(f"  - LoRA权重: {len(lora_keys)} 个")

    # 5. 验证和替换
    print(f"\n[5/6] 合并权重...")

    # 5.1 验证并替换Adapter（必须shape匹配）
    if adapter_keys:
        print("  验证Adapter形状...")
        validate_adapter_shapes(pretrained_state, trained_state, adapter_keys)
        print("  ✓ Adapter形状验证通过")

        for key in adapter_keys:
            pretrained_state[key] = trained_state[key].clone()
        print(f"  ✓ 已替换 {len(adapter_keys)} 个Adapter权重")

    # 5.2 处理LoRA权重（灵活模式）
    if lora_keys:
        if skip_lora_validation:
            print("  跳过LoRA形状验证（灵活模式）...")
            # 删除预训练模型中的旧LoRA权重
            old_lora_keys = [k for k in list(pretrained_state.keys()) if 'lora' in k.lower()]
            for k in old_lora_keys:
                del pretrained_state[k]
            print(f"  - 已删除预训练的 {len(old_lora_keys)} 个LoRA权重（旧rank）")

            # 添加新训练的LoRA权重
            for key in lora_keys:
                pretrained_state[key] = trained_state[key].clone()
            print(f"  ✓ 已添加 {len(lora_keys)} 个LoRA权重（新rank）")
        else:
            # 严格模式：验证shape
            print("  验证LoRA形状（严格模式）...")
            validate_adapter_shapes(pretrained_state, trained_state, lora_keys)
            for key in lora_keys:
                pretrained_state[key] = trained_state[key].clone()
            print(f"  ✓ 已替换 {len(lora_keys)} 个LoRA权重")

    # 6. 保存
    print(f"\n[6/6] 保存打包模型: {output_path}")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        'model_state_dict': pretrained_state,
        'args': pretrained_args,
    }, output_path)

    size_gb = output_path.stat().st_size / (1024 ** 3)
    print(f"  ✓ 保存成功！文件大小: {size_gb:.2f} GB")
    print("\n" + "=" * 60)
    print("打包完成！")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='灵活的模型打包工具（支持不同LoRA rank）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:

1. Stage 1 (仅Adapter):
  python training/scripts/package_model_with_lora_flexible.py \\
    --pretrained-model pretrained_models/FireRedASR-LLM-L/model.pth.tar \\
    --trained-weights exp/stage1/epoch-5.pt/pytorch_model.bin \\
    --output-model exp/stage1/packaged_model.pth.tar

2. Stage 2 (Adapter + LoRA, 使用不同rank):
  python training/scripts/package_model_with_lora_flexible.py \\
    --pretrained-model pretrained_models/FireRedASR-LLM-L/model.pth.tar \\
    --trained-weights exp/stage2/epoch-3.pt/pytorch_model.bin \\
    --output-model exp/stage2/packaged_model.pth.tar \\
    --skip-lora-validation  # 允许不同的LoRA rank
        """
    )

    parser.add_argument(
        '--pretrained-model',
        type=str,
        required=True,
        help='预训练模型路径 (model.pth.tar)'
    )

    parser.add_argument(
        '--trained-weights',
        type=str,
        required=True,
        help='训练后的权重路径 (epoch-N.pt/pytorch_model.bin)'
    )

    parser.add_argument(
        '--output-model',
        type=str,
        required=True,
        help='输出模型路径'
    )

    parser.add_argument(
        '--skip-lora-validation',
        action='store_true',
        help='跳过LoRA形状验证（当使用不同rank时必须开启）'
    )

    args = parser.parse_args()

    # 验证文件存在
    if not Path(args.pretrained_model).exists():
        raise FileNotFoundError(f"预训练模型不存在: {args.pretrained_model}")

    if not Path(args.trained_weights).exists():
        raise FileNotFoundError(f"训练权重不存在: {args.trained_weights}")

    # 执行打包
    merge_weights(
        args.pretrained_model,
        args.trained_weights,
        args.output_model,
        args.skip_lora_validation
    )


if __name__ == '__main__':
    main()
