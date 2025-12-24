# FireRedASR-LLM训练代码库

本目录包含FireRedASR-LLM模型的两阶段训练代码，结合了FireRedASR的Conformer编码器、Adapter投影层和Qwen2大语言模型。

## 架构概述

```
Audio → Conformer Encoder → Adapter → Qwen2-LLM → Text
         (冻结)              (可训练)   (Stage2: LoRA)
```

### 两阶段训练

- **Stage 1**: 仅训练Adapter投影层（编码器和LLM均冻结）
- **Stage 2**: 训练Adapter + LLM LoRA（编码器保持冻结）

## 环境配置

### 1. 安装依赖

```bash
# 基础依赖（已在FireRedASR的requirements.txt中）
pip install torch>=2.0.0
pip install transformers>=4.46.3
pip install peft>=0.13.2
pip install kaldi_native_fbank>=1.15
pip install sentencepiece

# 训练专用依赖
pip install deepspeed>=0.14.0
pip install lhotse
pip install tensorboard

# 注意：无需安装icefall，所有必要工具已集成在training/utils/中
```

### 2. 准备预训练模型

```bash
cd /path/to/FireRedASR

# 下载FireRedASR-AED-L模型
# 放置在 pretrained_models/FireRedASR-AED-L/
# 需要文件: model.pth.tar

# 下载Qwen2模型
# 推荐: Qwen2-7B-Instruct 或 Qwen2-1.5B-Instruct
# 放置在 pretrained_models/Qwen2-7B-Instruct/
```

### 3. 设置环境变量

```bash
export PATH=$PWD/fireredasr/:$PATH
export PYTHONPATH=$PWD/:$PYTHONPATH
```

## 数据准备

训练需要Lhotse格式的manifest文件，包含预计算的80维Fbank特征。

### 准备AISHELL-1数据（示例）

```python
from lhotse import CutSet, Fbank, FbankConfig
from lhotse.recipes import prepare_aishell

# 1. 准备manifest
manifests = prepare_aishell('/path/to/aishell')

# 2. 配置Fbank提取器（80维，与FireRedASR一致）
fbank_extractor = Fbank(FbankConfig(
    num_mel_bins=80,
    frame_length=0.025,
    frame_shift=0.01,
))

# 3. 提取并存储特征
cuts_train = CutSet.from_manifests(**manifests['train'])
cuts_train = cuts_train.compute_and_store_features(
    extractor=fbank_extractor,
    storage_path='data/fbank/aishell_feats',
    num_jobs=16
)
cuts_train.to_file('data/fbank/aishell_cuts_train.jsonl.gz')

# 重复处理dev和test集
cuts_dev = CutSet.from_manifests(**manifests['dev'])
cuts_dev = cuts_dev.compute_and_store_features(
    extractor=fbank_extractor,
    storage_path='data/fbank/aishell_feats',
    num_jobs=16
)
cuts_dev.to_file('data/fbank/aishell_cuts_dev.jsonl.gz')
```

### 快速测试数据集（用于代码测试）

如果您想快速测试训练流程，可以使用提供的脚本生成合成测试数据集：

```bash
# 生成100个合成样本（从单个音频文件重复生成）
python training/fireredasr_llm/prepare_test_data.py \
  --audio-path /path/to/audio.wav \
  --output-dir data/fbank/test_dataset \
  --num-samples 100
```

这将生成：
- `data/fbank/test_dataset/test_cuts_train.jsonl.gz` (80个样本)
- `data/fbank/test_dataset/test_cuts_dev.jsonl.gz` (9个样本)
- `data/fbank/test_dataset/test_cuts_test.jsonl.gz` (11个样本)
- `data/fbank/test_dataset/feats/` (预提取的80维Fbank特征)

使用测试数据集训练时，修改`train.py`中的数据加载代码：

```python
# 在train.py中取消注释以下行：
cuts_train = multi_dataset.test_dataset_train_cuts()
cuts_valid = multi_dataset.test_dataset_dev_cuts()
```

### MUSAN噪声数据（可选）

如果启用MUSAN噪声增强（`--enable-musan True`），需要准备MUSAN数据：

```bash
# 下载并准备MUSAN
# 生成 data/fbank/musan_cuts.jsonl.gz
```

## 训练

### Stage 1: 训练Adapter投影层

```bash
cd /path/to/FireRedASR

# 8卡训练
torchrun --nproc_per_node 8 training/fireredasr_llm/train.py \
  --training-stage 1 \
  --encoder-path pretrained_models/FireRedASR-AED-L/model.pth.tar \
  --llm-dir pretrained_models/Qwen2-7B-Instruct \
  --manifest-dir data/fbank \
  --exp-dir exp/fireredasr_llm_stage1 \
  --num-epochs 5 \
  --max-duration 200 \
  --num-workers 4 \
  --enable-spec-aug True \
  --enable-musan True \
  --use-fp16 True \
  --use-flash-attn True \
  --deepspeed \
  --deepspeed_config training/fireredasr_llm/ds_config_stage1.json
```

**Stage 1关键参数**:
- `--training-stage 1`: 指定为第一阶段
- `--max-duration 200`: 每批次最大音频时长（秒），可根据显存调整
- `--num-epochs 5`: 训练轮数
- `--enable-spec-aug True`: 启用SpecAugment数据增强
- `--enable-musan True`: 启用MUSAN噪声混合

### Stage 2: 联合训练Adapter + LoRA

```bash
# 使用Stage 1的最佳checkpoint作为初始化
torchrun --nproc_per_node 8 training/fireredasr_llm/train.py \
  --training-stage 2 \
  --encoder-path pretrained_models/FireRedASR-AED-L/model.pth.tar \
  --llm-dir pretrained_models/Qwen2-7B-Instruct \
  --stage1-checkpoint exp/fireredasr_llm_stage1/epoch-5.pt \
  --manifest-dir data/fbank \
  --exp-dir exp/fireredasr_llm_stage2 \
  --num-epochs 3 \
  --max-duration 100 \
  --num-workers 4 \
  --enable-spec-aug True \
  --enable-musan True \
  --use-fp16 True \
  --use-flash-attn True \
  --deepspeed \
  --deepspeed_config training/fireredasr_llm/ds_config_stage2.json
```

**Stage 2关键参数**:
- `--training-stage 2`: 指定为第二阶段
- `--stage1-checkpoint`: Stage 1的checkpoint路径（可选，用于初始化adapter）
- `--max-duration 100`: 由于LoRA开销，减小batch size
- `--num-epochs 3`: Stage 2通常需要较少轮数

**两阶段对比**:

| 阶段 | Encoder | Adapter | LLM | 训练参数 | 适用场景 |
|------|---------|---------|-----|----------|----------|
| Stage 1 | 冻结 | **训练** | 冻结 | ~22M | 快速对齐，低资源要求 |
| Stage 2 | 冻结 | **训练** | **LoRA** | ~161M | 联合优化，最佳性能（推荐） |

### 恢复训练

```bash
# 从特定epoch恢复
torchrun --nproc_per_node 8 training/fireredasr_llm/train.py \
  --training-stage 1 \
  --encoder-path pretrained_models/FireRedASR-AED-L/model.pth.tar \
  --llm-dir pretrained_models/Qwen2-7B-Instruct \
  --manifest-dir data/fbank \
  --exp-dir exp/fireredasr_llm_stage1 \
  --start-epoch 3 \
  --num-epochs 5 \
  --deepspeed \
  --deepspeed_config training/fireredasr_llm/ds_config_stage1.json
```

## 评估（待实现）

```bash
# 使用decode.py评估模型
python training/fireredasr_llm/decode.py \
  --checkpoint exp/fireredasr_llm_stage2/epoch-3.pt \
  --manifest-dir data/fbank \
  --test-set aishell_test
```

## Checkpoint管理

### Checkpoint文件结构

```
exp/fireredasr_llm_stage1/
├── epoch-1.pt              # FP32 state dict（可直接加载）
├── epoch-1-sampler.pt      # 采样器状态
├── epoch-1/                # DeepSpeed checkpoint（分布式训练用）
│   ├── zero_pp_rank_0_mp_rank_00_model_states.pt
│   └── ...
└── tensorboard/            # TensorBoard日志
```

### 加载Checkpoint进行推理

```python
from fireredasr.models.fireredasr_llm import FireRedAsrLlm
import torch

# 创建模型
args = argparse.Namespace(
    encoder_path="pretrained_models/FireRedASR-AED-L/model.pth.tar",
    llm_dir="pretrained_models/Qwen2-7B-Instruct",
    freeze_encoder=True,
    freeze_llm=False,
    use_lora=True,
    use_fp16=False,
    use_flash_attn=False,
    encoder_downsample_rate=2
)
model = FireRedAsrLlm.from_args(args)

# 加载训练好的weights
checkpoint = torch.load("exp/fireredasr_llm_stage2/epoch-3.pt")
model.load_state_dict(checkpoint, strict=False)
model.eval()

# 使用model.transcribe()进行推理
```

## 超参数建议

### Stage 1

| 参数 | 建议值 | 说明 |
|------|--------|------|
| num_epochs | 5 | 5-10轮通常足够 |
| max_duration | 200 | 8×V100可用200秒 |
| learning_rate | 1e-4 | DeepSpeed config中配置 |
| warmup_steps | 100 | DeepSpeed config中配置 |

### Stage 2

| 参数 | 建议值 | 说明 |
|------|--------|------|
| num_epochs | 3 | 3-5轮，避免过拟合 |
| max_duration | 100 | LoRA开销较大 |
| learning_rate | 1e-4 | 与Stage 1保持一致 |

## LoRA配置

默认LoRA配置（在`fireredasr/models/fireredasr_llm.py`中）:

```python
lora_config = LoraConfig(
    r=64,                 # LoRA秩
    lora_alpha=16,       # LoRA alpha
    target_modules=[     # 目标模块
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
        "up_proj", "gate_proj", "down_proj"      # MLP
    ],
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)
```

## 监控训练

### TensorBoard

```bash
tensorboard --logdir exp/fireredasr_llm_stage1/tensorboard --port 6006
```

### 日志文件

训练日志保存在 `exp/*/log-train-{rank}`

## 重要注意事项

### 1. 依赖要求

本训练代码库**已经集成了所有必要的icefall工具**，存放在`training/utils/`目录下：
- `utils/dist.py`：分布式训练工具（get_rank, get_world_size等）
- `utils/utils.py`：核心工具（AttributeDict, MetricsTracker, setup_logger等）

**无需单独安装icefall包**，所有依赖已经内置。

### 2. 环境依赖确认

训练前请确保已安装以下核心依赖：

```bash
# 必须依赖
pip install torch>=2.0.0
pip install transformers>=4.46.3
pip install deepspeed>=0.14.0
pip install peft>=0.13.2
pip install lhotse
pip install tensorboard

# FireRedASR依赖
pip install kaldi_native_fbank>=1.15
pip install sentencepiece
```

### 3. 数据准备是前提

**在开始训练之前必须完成Lhotse manifest准备**：

1. 将WAV文件转换为16kHz采样率
2. 提取80维Fbank特征
3. 生成Lhotse的CutSet manifest文件（.jsonl.gz格式）
4. 如果使用MUSAN，需要准备`musan_cuts.jsonl.gz`

示例数据准备脚本参见README中的"数据准备"章节。

### 4. DeepSpeed是必需的

本训练脚本**强制使用DeepSpeed**进行分布式训练，不支持纯PyTorch DDP。原因：
- FireRedASR-LLM模型较大（7B参数）
- DeepSpeed的ZeRO优化可节省显存
- 自动处理checkpoint转换

### 5. 两阶段训练顺序

**必须按顺序执行**：
1. Stage 1完成后，选择最佳checkpoint
2. Stage 2加载Stage 1的checkpoint继续训练
3. 不要跳过Stage 1直接训练Stage 2

## 常见问题

### 1. CUDA Out of Memory

**解决方案**:
- 减小`--max-duration`（如从200降到100）
- 减小`--num-workers`
- 使用更小的LLM（Qwen2-1.5B而非7B）
- 在DeepSpeed config中增大`gradient_accumulation_steps`
- 减少GPU数量，但保持total batch size不变

### 2. 验证loss不下降

**可能原因**:
- Stage 1训练不充分：增加epoch数到10轮
- Learning rate过大：在DeepSpeed config中降低学习率
- 数据质量问题：检查manifest文件和文本对齐
- 音频特征问题：确认Fbank提取正确（80维）

### 3. LoRA参数未更新

**检查**:
```python
# 查看可训练参数
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.shape)

# 应该看到包含'lora'的参数名
# 例如: llm.model.layers.0.self_attn.q_proj.lora_A.weight
```

如果没有LoRA参数，检查：
- `--training-stage 2`是否设置
- `use_lora=True`在model args中
- PEFT库是否正确安装

### 4. Input length限制

- **FireRedASR-AED编码器**：最大60秒（超出会出现hallucination）
- **FireRedASR-LLM**：推荐最大30秒
- 超出限制可能导致：
  - 位置编码错误
  - 生成重复文本
  - OOM错误

**建议**：在数据准备时过滤掉过长的音频。

### 5. 找不到manifest文件

**错误信息**：`FileNotFoundError: .../aishell_cuts_train.jsonl.gz`

**解决方案**：
- 确认`--manifest-dir`路径正确
- 检查manifest文件命名是否符合`multi_dataset.py`的要求
- 如果只用单个数据集，修改`train.py`中的数据加载代码

### 6. DeepSpeed checkpoint转换失败

**错误信息**：ZeRO checkpoint conversion error

**解决方案**：
- 确保所有rank的checkpoint文件都完整
- 使用DeepSpeed提供的转换工具手动转换：
  ```bash
  python -m deepspeed.utils.zero_to_fp32 \
    exp/stage1/epoch-5/ \
    exp/stage1/epoch-5.pt \
    --tag epoch-5 \
    --exclude_frozen_parameters
  ```

### 7. 导入错误

**错误信息**：`ModuleNotFoundError: No module named 'utils'`

**解决方案**：
- 确保从`training/fireredasr_llm/`目录运行脚本
- 或者设置PYTHONPATH：
  ```bash
  export PYTHONPATH=/path/to/FireRedASR/training:$PYTHONPATH
  ```

## 数据集支持

`multi_dataset.py`支持以下中文ASR数据集：

- AISHELL-1/2/4
- THCHS-30
- ST-CMDS
- Primewords
- MagicData
- Ali-Meeting
- WeNetSpeech
- KeSpeech

### 使用单个数据集

修改`train.py`中的数据加载代码：

```python
# 仅使用AISHELL-1
cuts_train = multi_dataset.aishell_train_cuts()
cuts_valid = multi_dataset.aishell_dev_cuts()

# 使用全部数据集
cuts_train = multi_dataset.train_cuts()
cuts_valid = multi_dataset.dev_cuts()
```

## 性能预期

在8×V100 GPU上的预估训练时间：

- **Stage 1** (AISHELL-1 only): ~2-3小时/epoch
- **Stage 2** (AISHELL-1 only): ~3-4小时/epoch
- **Full multi-dataset**: 10-20倍时间

## 参考

- [FireRedASR GitHub](https://github.com/FireRedTeam/FireRedASR)
- [Icefall ASR_LLM](https://github.com/k2-fsa/icefall/tree/master/egs/speech_llm/ASR_LLM)
- [Qwen2 Models](https://huggingface.co/Qwen)
- [Lhotse Documentation](https://lhotse.readthedocs.io/)

## 文件说明

- `train.py`: 主训练脚本
- `asr_datamodule.py`: Lhotse数据加载模块
- `multi_dataset.py`: 多数据集管理
- `ds_config_stage1.json`: Stage 1 DeepSpeed配置
- `ds_config_stage2.json`: Stage 2 DeepSpeed配置
- `decode.py`: 评估脚本（完整实现，支持beam search和CER计算）
- `prepare_test_data.py`: 测试数据集生成脚本
- `README.md`: 本文档

## License

Apache License 2.0
