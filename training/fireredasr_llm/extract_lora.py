#!/usr/bin/env python3
"""
提取LoRA权重脚本

从完整的Stage 2 checkpoint中提取：
1. LoRA参数（单独保存）
2. Adapter参数（单独保存）
3. 生成HuggingFace PEFT格式的adapter_config.json

Usage:
    python extract_lora.py \
        --checkpoint exp/fireredasr_llm_stage2/epoch-3.pt/pytorch_model.bin \
        --output-dir exp/fireredasr_llm_stage2/lora_weights
"""

import argparse
import json
import torch
from pathlib import Path


def extract_lora_weights(checkpoint_path: str, output_dir: str):
    """从checkpoint中提取LoRA和Adapter权重"""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"加载checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # 分类参数
    lora_params = {k: v for k, v in ckpt.items() if 'lora' in k.lower()}
    adapter_params = {k: v for k, v in ckpt.items()
                     if 'encoder_projector' in k or 'speech_projector' in k}

    # 统计信息
    print(f"\n{'='*60}")
    print(f"参数统计:")
    print(f"{'='*60}")

    lora_size = sum(p.numel() for p in lora_params.values())
    print(f"LoRA参数:")
    print(f"  - 数量: {len(lora_params)}")
    print(f"  - 总元素: {lora_size:,}")
    print(f"  - 存储大小: {lora_size * 4 / 1024 / 1024:.1f} MB (FP32)")

    adapter_size = sum(p.numel() for p in adapter_params.values())
    print(f"\nAdapter参数:")
    print(f"  - 数量: {len(adapter_params)}")
    print(f"  - 总元素: {adapter_size:,}")
    print(f"  - 存储大小: {adapter_size * 4 / 1024 / 1024:.1f} MB (FP32)")

    # 分析LoRA结构
    layers = set()
    module_types = set()
    for name in lora_params.keys():
        if 'layers.' in name:
            layer_num = name.split('layers.')[1].split('.')[0]
            layers.add(int(layer_num))
        if 'lora_A' in name:
            module = name.split('lora_A')[0].split('.')[-2]
            module_types.add(module)

    print(f"\nLoRA结构:")
    print(f"  - 应用层数: {len(layers)} (Qwen2-7B有28层)")
    print(f"  - 层编号: {sorted(layers)}")
    print(f"  - 模块类型: {sorted(module_types)}")

    # 提取rank信息
    sample_lora_A = None
    sample_lora_B = None
    for name, param in lora_params.items():
        if 'lora_A' in name:
            sample_lora_A = param
        if 'lora_B' in name:
            sample_lora_B = param
            break

    if sample_lora_A is not None and sample_lora_B is not None:
        rank = sample_lora_A.shape[0]
        print(f"  - LoRA rank: {rank}")
        print(f"  - LoRA alpha: 推断为 {rank // 4} (假设alpha=rank/4)")

    # 保存LoRA权重
    lora_path = output_dir / "lora_weights.pt"
    print(f"\n保存LoRA权重到: {lora_path}")
    torch.save(lora_params, lora_path)

    # 保存Adapter权重
    adapter_path = output_dir / "adapter_weights.pt"
    print(f"保存Adapter权重到: {adapter_path}")
    torch.save(adapter_params, adapter_path)

    # 保存完整权重（LoRA + Adapter）
    combined_path = output_dir / "combined_weights.pt"
    combined = {**lora_params, **adapter_params}
    print(f"保存组合权重到: {combined_path}")
    torch.save(combined, combined_path)

    # 生成HuggingFace PEFT格式的adapter_config.json
    adapter_config = {
        "auto_mapping": None,
        "base_model_name_or_path": "Qwen/Qwen2-7B-Instruct",
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "layers_pattern": None,
        "layers_to_transform": list(sorted(layers)),
        "lora_alpha": rank // 4 if sample_lora_A is not None else 16,
        "lora_dropout": 0.0,
        "megatron_config": None,
        "megatron_core": "megatron.core",
        "modules_to_save": None,
        "peft_type": "LORA",
        "r": rank if sample_lora_A is not None else 64,
        "rank_pattern": {},
        "revision": None,
        "target_modules": sorted(module_types),
        "task_type": None,
        "use_dora": False,
        "use_rslora": False
    }

    config_path = output_dir / "adapter_config.json"
    print(f"保存PEFT配置到: {config_path}")
    with open(config_path, 'w') as f:
        json.dump(adapter_config, f, indent=2)

    # 保存参数名列表（用于调试）
    params_list = {
        "lora_params": sorted(lora_params.keys()),
        "adapter_params": sorted(adapter_params.keys())
    }
    params_list_path = output_dir / "params_list.json"
    print(f"保存参数列表到: {params_list_path}")
    with open(params_list_path, 'w') as f:
        json.dump(params_list, f, indent=2)

    print(f"\n{'='*60}")
    print(f"提取完成！")
    print(f"{'='*60}")
    print(f"\n输出文件:")
    print(f"  {lora_path} - LoRA权重 ({lora_size * 4 / 1024 / 1024:.1f} MB)")
    print(f"  {adapter_path} - Adapter权重 ({adapter_size * 4 / 1024 / 1024:.1f} MB)")
    print(f"  {combined_path} - 组合权重 ({(lora_size + adapter_size) * 4 / 1024 / 1024:.1f} MB)")
    print(f"  {config_path} - PEFT配置")
    print(f"  {params_list_path} - 参数名列表")

    return lora_params, adapter_params


def main():
    parser = argparse.ArgumentParser(description="提取LoRA和Adapter权重")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="完整checkpoint路径 (pytorch_model.bin)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="输出目录"
    )

    args = parser.parse_args()

    extract_lora_weights(args.checkpoint, args.output_dir)


if __name__ == "__main__":
    main()
