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

## 评估与测试

### 快速测试

使用测试脚本快速验证训练效果：

```bash
# 使用提供的测试脚本（测试合成数据集）
bash test_decode.sh

# 或手动运行decode.py
python training/fireredasr_llm/decode.py \
  --checkpoint exp/test_stage2/epoch-1.pt/pytorch_model.bin \
  --encoder-path pretrained_models/FireRedASR-LLM-L/asr_encoder.pth.tar \
  --llm-dir pretrained_models/Qwen2-7B-Instruct \
  --manifest-dir data/fbank \
  --test-manifest test_dataset/test_cuts_test.jsonl.gz \
  --output-dir results/test_stage2 \
  --beam-size 3 \
  --repetition-penalty 3.0 \
  --use-gpu 1
```

### 在AISHELL-1测试集上评估

```bash
python training/fireredasr_llm/decode.py \
  --checkpoint exp/fireredasr_llm_stage2/epoch-3.pt/pytorch_model.bin \
  --encoder-path pretrained_models/FireRedASR-AED-L/model.pth.tar \
  --llm-dir pretrained_models/Qwen2-7B-Instruct \
  --manifest-dir data/fbank \
  --test-manifest aishell_cuts_test.jsonl.gz \
  --output-dir results/aishell_test \
  --beam-size 3 \
  --repetition-penalty 3.0 \
  --llm-length-penalty 1.0 \
  --temperature 1.0 \
  --max-duration 100.0 \
  --use-gpu 1
```

### 解码参数调整

不同场景下可以调整以下参数：

| 参数 | 默认值 | 说明 | 建议 |
|------|--------|------|------|
| `--beam-size` | 3 | Beam search大小 | 1(greedy/快速), 3(平衡), 5-10(高质量) |
| `--repetition-penalty` | 3.0 | 重复惩罚 | 1.0(无), 3.0(推荐), 5.0(强惩罚) |
| `--llm-length-penalty` | 1.0 | 长度惩罚 | <1.0(短句), 1.0(中性), >1.0(长句) |
| `--temperature` | 1.0 | 采样温度 | 仅在sampling时有效 |
| `--max-duration` | 100.0 | 批次大小(秒) | 根据显存调整 |

### 查看评估结果

```bash
# 查看CER和详细结果
cat results/aishell_test/aishell_cuts_test.jsonl_results.txt

# 预期结果（完整训练后）
# CER: < 10% (良好)
# CER: < 5%  (优秀)
# CER: < 2%  (接近完美)
```

**详细测试文档**: 参见项目根目录的 `TESTING_GUIDE.md`

## Checkpoint管理与模型打包

### Checkpoint文件结构

训练过程会自动生成以下文件：

```
exp/fireredasr_llm_stage2/
├── epoch-1.pt/                         # FP32 checkpoint（推荐使用）
│   └── pytorch_model.bin               # 完整模型权重 (749 MB)
├── epoch-1-sampler.pt                  # 采样器状态
├── epoch-1/                            # DeepSpeed原始checkpoint (可删除)
│   ├── mp_rank_00_model_states.pt      # 模型状态 (750 MB)
│   └── zero_pp_rank_0_mp_rank_00_optim_states.pt  # 优化器状态 (2.1 GB)
├── latest                              # 指向最新checkpoint
└── tensorboard/                        # TensorBoard日志
```

**推荐保留**:
- ✅ `epoch-N.pt/pytorch_model.bin` - 用于推理和部署
- ✅ `tensorboard/` - 训练曲线可视化
- ❌ `epoch-N/` - 原始DeepSpeed checkpoint（已转换，可删除节省空间）

### 提取LoRA权重（可选）

如果需要单独管理LoRA权重或使用HuggingFace PEFT格式：

```bash
python training/fireredasr_llm/extract_lora.py \
  --checkpoint exp/fireredasr_llm_stage2/epoch-3.pt/pytorch_model.bin \
  --output-dir exp/fireredasr_llm_stage2/lora_weights
```

生成的文件：
```
lora_weights/
├── lora_weights.pt            # 仅LoRA参数 (616 MB)
├── adapter_weights.pt         # 仅Adapter参数 (84 MB)
├── combined_weights.pt        # LoRA + Adapter (700 MB)
├── adapter_config.json        # HuggingFace PEFT配置
└── params_list.json           # 参数名称列表
```

### 加载Checkpoint进行推理

#### 方法1: 直接加载完整checkpoint

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

# 加载训练好的权重
checkpoint = torch.load(
    "exp/fireredasr_llm_stage2/epoch-3.pt/pytorch_model.bin",
    map_location="cpu",
    weights_only=False
)
model.load_state_dict(checkpoint, strict=False)
model.eval()

# 推理
results = model.transcribe(
    audio_path="test.wav",
    beam_size=3,
    repetition_penalty=3.0
)
print(results)
```

#### 方法2: 使用HuggingFace PEFT加载LoRA

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

# 加载base model
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-7B-Instruct")

# 加载LoRA adapter
model = PeftModel.from_pretrained(
    base_model,
    "exp/fireredasr_llm_stage2/lora_weights",
    is_trainable=False
)
model.eval()

# 注意：这种方法只加载LLM的LoRA，不包括Encoder和Adapter
# 需要配合FireRedASR的encoder使用
```

## 模型打包与部署

### 1. 打包为可发布的模型

创建一个完整的模型目录，包含所有必要文件：

```bash
# 创建模型目录
mkdir -p my_fireredasr_llm_model

# 复制必要文件
cp -r pretrained_models/FireRedASR-LLM-L/* my_fireredasr_llm_model/

# 替换训练的模型权重
cp exp/fireredasr_llm_stage2/epoch-3.pt/pytorch_model.bin \
   my_fireredasr_llm_model/model.pth.tar

# 打包
tar -czf my_fireredasr_llm_model.tar.gz my_fireredasr_llm_model/
```

生成的目录结构：
```
my_fireredasr_llm_model/
├── model.pth.tar              # 训练的模型权重 (749 MB)
├── asr_encoder.pth.tar        # ASR encoder权重 (2.7 GB)
├── cmvn.ark                   # CMVN统计量
├── Qwen2-7B-Instruct/         # Qwen2 base model (软链接或拷贝)
│   ├── config.json
│   ├── model-00001-of-00004.safetensors
│   ├── ...
│   └── tokenizer.json
└── README.md                  # 模型说明文档
```

### 2. 创建模型说明文档

为打包的模型创建README：

```bash
cat > my_fireredasr_llm_model/README.md << 'EOF'
# Custom FireRedASR-LLM Model

## 模型信息

- **训练数据**: AISHELL-1 (178小时中文语音)
- **训练轮数**: Stage 1 (5 epochs) + Stage 2 (3 epochs)
- **架构**: Conformer Encoder + Adapter + Qwen2-7B-Instruct (LoRA)
- **性能**: CER ~X.X% on AISHELL-1 test set

## 使用方法

### 安装依赖
```bash
pip install torch transformers peft kaldi_native_fbank sentencepiece
```

### 推理示例
```python
from fireredasr.models.fireredasr import FireRedAsr

# 加载模型
model = FireRedAsr.from_pretrained(
    model_dir="my_fireredasr_llm_model",
    asr_type="llm"
)

# 识别音频
result = model.transcribe(
    audio_path="test.wav",
    beam_size=3,
    repetition_penalty=3.0
)
print(result)
```

## 训练信息

- 训练时间: ~XX小时 on 8x V100
- 最终训练loss: X.XXX
- 最终验证loss: X.XXX
- 训练日期: YYYY-MM-DD

## 许可证

Apache License 2.0
EOF
```

### 3. 模型版本管理

建议使用语义化版本号管理模型：

```bash
# 版本号格式: v{major}.{minor}.{patch}
# major: 架构变更
# minor: 训练数据或超参数变更
# patch: 小修复或微调

# 示例
my_fireredasr_llm_model_v1.0.0.tar.gz  # 初始版本
my_fireredasr_llm_model_v1.1.0.tar.gz  # 添加更多训练数据
my_fireredasr_llm_model_v1.1.1.tar.gz  # 修复解码bug
```

### 4. 上传到HuggingFace Hub（可选）

将模型上传到HuggingFace以便分享：

```python
from huggingface_hub import HfApi

api = HfApi()

# 上传模型
api.upload_folder(
    folder_path="my_fireredasr_llm_model",
    repo_id="your-username/fireredasr-llm-aishell",
    repo_type="model",
)
```

### 5. 模型量化（可选）

减小模型大小以便部署：

```python
import torch
from transformers import AutoModelForCausalLM

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    "pretrained_models/Qwen2-7B-Instruct",
    device_map="auto",
    torch_dtype=torch.float16  # FP16量化
)

# 或使用4-bit量化（需要bitsandbytes）
model = AutoModelForCausalLM.from_pretrained(
    "pretrained_models/Qwen2-7B-Instruct",
    device_map="auto",
    load_in_4bit=True,  # 4-bit量化
    bnb_4bit_compute_dtype=torch.float16
)
```

### 6. 部署为API服务

使用FastAPI创建REST API：

```python
# server.py
from fastapi import FastAPI, File, UploadFile
from fireredasr.models.fireredasr import FireRedAsr
import tempfile

app = FastAPI()
model = FireRedAsr.from_pretrained("my_fireredasr_llm_model", asr_type="llm")

@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    # 保存上传的音频
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

    # 识别
    result = model.transcribe(tmp_path, beam_size=3)

    return {"text": result}

# 运行: uvicorn server:app --host 0.0.0.0 --port 8000
```

### 7. Docker部署（推荐）

创建Dockerfile：

```dockerfile
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install -r requirements.txt

# 复制模型和代码
COPY my_fireredasr_llm_model/ /app/models/
COPY fireredasr/ /app/fireredasr/
COPY server.py /app/

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
```

构建和运行：
```bash
# 构建镜像
docker build -t fireredasr-llm-api:v1.0.0 .

# 运行容器
docker run -d --gpus all -p 8000:8000 fireredasr-llm-api:v1.0.0

# 测试API
curl -X POST -F "audio=@test.wav" http://localhost:8000/transcribe
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

### 核心训练文件

- `train.py`: 主训练脚本，支持两阶段训练和分布式训练
- `asr_datamodule.py`: Lhotse数据加载模块，支持动态bucketing
- `multi_dataset.py`: 多数据集管理，支持AISHELL、WenetSpeech等
- `ds_config_stage1.json`: Stage 1 DeepSpeed配置 (FP16, ZeRO-1)
- `ds_config_stage2.json`: Stage 2 DeepSpeed配置 (FP16, ZeRO-1)
- `ds_config_stage2_fp32.json`: Stage 2 FP32配置 (测试用)

### 评估和工具

- `decode.py`: 完整评估脚本，支持beam search、CER/WER计算
- `extract_lora.py`: LoRA权重提取工具，生成HuggingFace PEFT格式
- `prepare_test_data.py`: 合成测试数据集生成脚本

### 测试脚本

- `test_stage1_training.sh`: Stage 1快速测试脚本
- `test_stage2_training.sh`: Stage 2快速测试脚本
- `test_decode.sh`: 模型识别效果测试脚本

### 文档

- `README.md`: 本文档（训练、评估、部署完整指南）
- `../../TESTING_GUIDE.md`: 详细测试指南
- `../../ALL_STAGES_TEST_REPORT.md`: 完整测试报告

## License

Apache License 2.0
