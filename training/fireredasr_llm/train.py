#!/usr/bin/env python3
# Training script for FireRedASR-LLM
# Adapted from icefall whisper_llm_zh/train.py
#
# Usage:
# Stage 1 (train projector only):
# torchrun --nproc_per_node 8 training/fireredasr_llm/train.py \
#   --training-stage 1 \
#   --encoder-path pretrained_models/FireRedASR-AED-L/model.pth.tar \
#   --llm-dir pretrained_models/Qwen2-7B-Instruct \
#   --manifest-dir data/fbank \
#   --exp-dir exp/fireredasr_llm_stage1 \
#   --deepspeed \
#   --deepspeed_config training/fireredasr_llm/ds_config_stage1.json
#
# Stage 2 (train projector + LoRA):
# torchrun --nproc_per_node 8 training/fireredasr_llm/train.py \
#   --training-stage 2 \
#   --encoder-path pretrained_models/FireRedASR-AED-L/model.pth.tar \
#   --llm-dir pretrained_models/Qwen2-7B-Instruct \
#   --stage1-checkpoint exp/fireredasr_llm_stage1/epoch-5.pt \
#   --manifest-dir data/fbank \
#   --exp-dir exp/fireredasr_llm_stage2 \
#   --deepspeed \
#   --deepspeed_config training/fireredasr_llm/ds_config_stage2.json

import argparse
import copy
import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import deepspeed
import torch
import torch.nn as nn
import transformers
from asr_datamodule import AsrDataModule
from deepspeed.utils.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict
from lhotse.utils import fix_random_seed
from multi_dataset import MultiDataset
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

# Import from FireRedASR
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from fireredasr.models.fireredasr_llm import FireRedAsrLlm
from fireredasr.tokenizer.llm_tokenizer import (
    LlmTokenizerWrapper,
    DEFAULT_SPEECH_TOKEN,
    IGNORE_TOKEN_ID
)

# Import training utilities
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.dist import get_rank, get_world_size
from utils.utils import AttributeDict, MetricsTracker, setup_logger, str2bool


def add_model_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--training-stage",
        type=int,
        required=True,
        choices=[1, 2],
        help="Training stage: 1=projector only, 2=projector+LoRA",
    )
    parser.add_argument(
        "--encoder-path",
        type=str,
        required=True,
        help="Path to FireRedASR-AED checkpoint (model.pth.tar)",
    )
    parser.add_argument(
        "--llm-dir",
        type=str,
        required=True,
        help="Path to Qwen2 model directory",
    )
    parser.add_argument(
        "--stage1-checkpoint",
        type=str,
        default="",
        help="Path to Stage 1 checkpoint (for Stage 2 training)",
    )
    parser.add_argument(
        "--use-fp16",
        type=str2bool,
        default=True,
        help="Use FP16 mixed precision training",
    )
    parser.add_argument(
        "--use-flash-attn",
        type=str2bool,
        default=True,
        help="Use Flash Attention 2",
    )
    parser.add_argument(
        "--pretrained-model",
        type=str,
        default="",
        help="Path to pretrained model (for adapter initialization). "
             "If provided, adapter weights will be loaded from this checkpoint.",
    )


def add_training_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--exp-dir",
        type=Path,
        default=Path("exp/fireredasr_llm"),
        help="Directory to save checkpoints and logs",
    )
    parser.add_argument(
        "--start-epoch",
        type=int,
        default=1,
        help="Resume training from this epoch",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=5,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--valid-interval",
        type=int,
        default=5000,
        help="Run validation every N batches",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=50,
        help="Log metrics every N batches",
    )
    parser.add_argument(
        "--tensorboard",
        type=str2bool,
        default=True,
        help="Use tensorboard for logging",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--use-test-dataset",
        type=str2bool,
        default=True,
        help="Use synthetic test dataset (True) or real dataset (False)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="aishell",
        choices=["aishell", "all", "test", "custom"],
        help="Which real dataset to use: aishell (single), all (multi-dataset), test (synthetic), or custom (urls.txt + ref.txt)",
    )
    # DeepSpeed arguments
    parser.add_argument(
        "--deepspeed",
        action="store_true",
        help="Use DeepSpeed for training",
    )
    parser.add_argument(
        "--deepspeed_config",
        type=str,
        default="",
        help="Path to DeepSpeed config JSON",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training",
    )


def get_parser():
    parser = argparse.ArgumentParser(
        description="Train FireRedASR-LLM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    add_model_arguments(parser)
    add_training_arguments(parser)
    AsrDataModule.add_arguments(parser)

    return parser


def get_params() -> AttributeDict:
    """Return training parameters"""
    params = AttributeDict(
        {
            "best_train_loss": float("inf"),
            "best_valid_loss": float("inf"),
            "best_train_epoch": -1,
            "best_valid_epoch": -1,
            "batch_idx_train": 0,
            "cur_epoch": 1,
        }
    )
    return params


def build_model(params: AttributeDict) -> FireRedAsrLlm:
    """
    Build FireRedASR-LLM model based on training stage.

    Stage 1: freeze_encoder=True, freeze_llm=True, use_lora=False
    Stage 2: freeze_encoder=True, freeze_llm=False, use_lora=True
    """
    logging.info(f"Building model for Stage {params.training_stage}")

    # Create model arguments
    args = argparse.Namespace()
    args.encoder_path = params.encoder_path
    args.llm_dir = params.llm_dir
    args.freeze_encoder = True  # Always freeze encoder
    args.use_fp16 = params.use_fp16
    args.use_flash_attn = params.use_flash_attn
    args.encoder_downsample_rate = 2  # FireRedASR uses 2x

    if params.training_stage == 1:
        # Stage 1: Train projector only
        args.freeze_llm = True
        args.use_lora = False
        logging.info("Stage 1: Training projector only (encoder and LLM frozen)")
    else:
        # Stage 2: Train projector + LoRA
        args.freeze_llm = False
        args.use_lora = True
        logging.info("Stage 2: Training projector + LoRA (encoder frozen)")

    # Build model
    model = FireRedAsrLlm.from_args(args)

    # Convert encoder to FP32 if not using FP16
    if not params.use_fp16:
        logging.info("Converting encoder to FP32")
        model.encoder = model.encoder.float()
        logging.info("Encoder converted to FP32")

    if params.pretrained_model:
        print(f"\n{'='*80}", flush=True)
        print(f"[ADAPTER LOADING] Starting adapter loading from: {params.pretrained_model}", flush=True)
        print(f"{'='*80}\n", flush=True)
        logging.info(f"Loading pretrained weights from {params.pretrained_model}")
        try:
            pretrained = torch.load(params.pretrained_model, map_location="cpu", weights_only=False)
            pretrained_state = pretrained['model_state_dict']
            print(f"[ADAPTER LOADING] Pretrained model loaded, total keys: {len(pretrained_state)}", flush=True)


            # Extract adapter weights
            adapter_keys = {k: v for k, v in pretrained_state.items()
                           if k.startswith('encoder_projector')}

            if adapter_keys:
                print(f"[ADAPTER LOADING] Found {len(adapter_keys)} adapter keys", flush=True)
                missing, unexpected = model.load_state_dict(adapter_keys, strict=False)
                print(f"[ADAPTER LOADING] ✅ Successfully loaded {len(adapter_keys)} adapter keys!", flush=True)
                logging.info(f"✅ Loaded {len(adapter_keys)} adapter keys from pretrained model")
                for k in adapter_keys.keys():
                    print(f"[ADAPTER LOADING]   - {k}", flush=True)
                    logging.info(f"  - {k}")
                print(f"{'='*80}\n", flush=True)
            else:
                print(f"[ADAPTER LOADING] ⚠️ WARNING: No adapter keys found in pretrained model!", flush=True)
                logging.warning("⚠️ No adapter keys found in pretrained model!")

            # Check for LoRA weights in pretrained model
            # NOTE: Pretrained models should have LoRA weights pre-merged into base LLM offline.
            # Use training/scripts/merge_lora_to_base.py to merge before training.
            # Only adapter weights will be loaded from pretrained checkpoints.
            lora_keys = [k for k in pretrained_state.keys() if 'lora' in k.lower()]
            if lora_keys:
                print(f"[LORA WARNING] ⚠️ Found {len(lora_keys)} LoRA keys in pretrained model!", flush=True)
                print(f"[LORA WARNING] Pretrained LoRA weights will NOT be loaded (merge offline first)", flush=True)
                print(f"[LORA WARNING] Use: python training/scripts/merge_lora_to_base.py", flush=True)
                logging.warning(f"⚠️ Found {len(lora_keys)} LoRA keys in pretrained model!")
                logging.warning("Pretrained LoRA weights will NOT be loaded (merge offline first)")
                logging.warning("Recommendation: Use training/scripts/merge_lora_to_base.py to merge LoRA into base LLM")
            else:
                print(f"[LORA INFO] ℹ️ No LoRA keys in pretrained model (expected)", flush=True)
                logging.info("ℹ️ No LoRA keys in pretrained model (clean adapter checkpoint)")

        except Exception as e:
            print(f"[ADAPTER LOADING] ❌ ERROR: {e}", flush=True)
            logging.error(f"❌ Failed to load pretrained model: {e}")
            logging.error("Continuing with random adapter initialization...")
    else:
        print(f"\n[ADAPTER LOADING] ⚠️ WARNING: No --pretrained-model specified!", flush=True)
        print(f"[ADAPTER LOADING] Adapter will be randomly initialized!\n", flush=True)
        logging.warning("⚠️ No pretrained model specified - adapter will be randomly initialized!")
        logging.warning("⚠️ This may lead to poor performance. Recommend using --pretrained-model")

    # Load Stage 1 checkpoint for Stage 2
    if params.training_stage == 2 and params.stage1_checkpoint:
        print(f"\n{'='*80}", flush=True)
        print(f"[STAGE1 CHECKPOINT] Loading from: {params.stage1_checkpoint}", flush=True)
        print(f"{'='*80}\n", flush=True)
        logging.info(f"Loading Stage 1 checkpoint: {params.stage1_checkpoint}")

        checkpoint = torch.load(params.stage1_checkpoint, map_location="cpu", weights_only=False)

        # Validate checkpoint structure
        adapter_keys = [k for k in checkpoint.keys() if 'encoder_projector' in k]
        lora_keys = [k for k in checkpoint.keys() if 'lora' in k.lower()]

        print(f"[STAGE1 CHECKPOINT] Checkpoint contains {len(adapter_keys)} adapter keys, {len(lora_keys)} LoRA keys", flush=True)
        logging.info(f"Stage 1 checkpoint contains {len(adapter_keys)} adapter keys, {len(lora_keys)} LoRA keys")

        if lora_keys:
            print(f"[STAGE1 CHECKPOINT] ⚠️ WARNING: Stage 1 checkpoint should not contain LoRA weights!", flush=True)
            logging.warning("⚠️ Stage 1 checkpoint should not contain LoRA weights!")

        # Load adapter weights
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)

        # Filter out expected missing keys (LoRA is expected to be missing from Stage 1)
        unexpected_missing = [k for k in missing_keys if 'lora' not in k.lower()]
        if unexpected_missing:
            print(f"[STAGE1 CHECKPOINT] ❌ Unexpected missing keys (non-LoRA): {unexpected_missing[:10]}", flush=True)
            logging.error(f"Unexpected missing keys (non-LoRA): {unexpected_missing[:10]}")

        print(f"[STAGE1 CHECKPOINT] ✅ Loaded adapter weights from Stage 1 checkpoint", flush=True)
        print(f"[STAGE1 CHECKPOINT] ℹ️ LoRA weights initialized randomly (expected for Stage 2 start)", flush=True)
        logging.info(f"✅ Loaded adapter weights from Stage 1 checkpoint")
        logging.info(f"ℹ️ LoRA weights initialized randomly (expected for Stage 2 start)")
        print(f"{'='*80}\n", flush=True)

    # Log trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} "
                 f"({100 * trainable_params / total_params:.2f}%)")

    return model


def validate_chat_template(template, stage="training"):
    """
    Validate chat template structure to ensure training/inference consistency.

    This validation ensures that:
    1. Template includes required role markers (<|im_start|>, user, assistant)
    2. Training template ends with <|im_end|> (model should predict EOS)
    3. Inference template does NOT end with <|im_end|> (model generates from right position)

    This difference is intentional and matches icefall's design.
    """
    from jinja2 import Template as JinjaTemplate

    test_messages = [
        {"role": "user", "content": "<speech>请转写音频为文字"},
        {"role": "assistant", "content": "测试文本"}
    ]

    try:
        jinja_template = JinjaTemplate(template)
        result = jinja_template.render(messages=test_messages)

        # Check structure
        assert '<|im_start|>user' in result, "Missing user role marker"
        assert '<|im_start|>assistant' in result, "Missing assistant role marker"
        assert '<speech>请转写音频为文字' in result, "Missing speech token"

        # Training should end with <|im_end|>, inference should not
        if stage == "training":
            assert result.endswith('<|im_end|>'), "Training template should end with <|im_end|>"
            logging.info(f"✅ {stage.capitalize()} template validation passed")
        else:
            assert not result.endswith('<|im_end|>'), "Inference template should NOT end with <|im_end|>"
            logging.info(f"✅ {stage.capitalize()} template validation passed")

        logging.debug(f"Template output:\n{result}")
        return True

    except Exception as e:
        logging.error(f"❌ Template validation failed: {e}")
        return False


def preprocess_texts(
    messages,
    tokenizer: transformers.PreTrainedTokenizer,
    max_len: int = 128,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Preprocess text messages for training.

    Returns:
        input_ids: (batch_size, seq_len)
        attention_mask: (batch_size, seq_len)
        target_ids: (batch_size, seq_len) with prompts masked using IGNORE_TOKEN_ID
    """
    texts = []
    # Training template: includes <|im_end|> at the end so model learns to predict EOS
    # This differs from inference template (which omits <|im_end|>) - see fireredasr/tokenizer/llm_tokenizer.py
    # This design matches icefall's approach and is intentional
    TEMPLATE = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content']}}{% if loop.last %}{{ '<|im_end|>'}}{% else %}{{ '<|im_end|>\n' }}{% endif %}{% endfor %}"

    for i, msg in enumerate(messages):
        texts.append(
            tokenizer.apply_chat_template(
                msg,
                tokenize=True,
                chat_template=TEMPLATE,
                add_generation_prompt=False,
                padding="longest",
                max_length=max_len,
                truncation=True,
            )
        )

    # Pad texts to same length
    max_len_texts = max([len(text) for text in texts])
    if tokenizer.padding_side == "right":
        texts = [
            text + [tokenizer.pad_token_id] * (max_len_texts - len(text))
            for text in texts
        ]
    else:
        texts = [
            [tokenizer.pad_token_id] * (max_len_texts - len(text)) + text
            for text in texts
        ]

    input_ids = torch.tensor(texts, dtype=torch.long)
    target_ids = input_ids.clone()

    # Mask padding tokens
    target_ids[target_ids == tokenizer.pad_token_id] = IGNORE_TOKEN_ID

    # Mask prompt tokens (before "assistant")
    mask_indices = torch.where(
        input_ids == tokenizer.convert_tokens_to_ids("assistant")
    )
    for i in range(mask_indices[0].size(0)):
        row = mask_indices[0][i]
        col = mask_indices[1][i]
        # +2 to skip 'assistant' and '\n'
        target_ids[row, : col + 2] = IGNORE_TOKEN_ID

    attention_mask = input_ids.ne(tokenizer.pad_token_id)

    return input_ids, attention_mask, target_ids


def compute_loss(
    params: AttributeDict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: nn.Module,
    batch: dict,
    is_training: bool,
) -> Tuple[Tensor, MetricsTracker]:
    """
    Compute loss for a batch.

    Args:
        params: Training parameters
        tokenizer: Tokenizer
        model: Model
        batch: Batch from DataLoader
        is_training: Whether in training mode

    Returns:
        loss: Scalar loss
        info: MetricsTracker with loss info
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    # Get features (N, T, F)
    feature = batch["inputs"]
    assert feature.ndim == 3
    feature = feature.to(device=device, dtype=dtype)

    # Get text supervisions
    texts = batch["supervisions"]["text"]
    feature_lengths = batch["supervisions"]["num_frames"].to(device)

    # Prepare messages in chat format
    messages = []
    for text in texts:
        text = text.replace(" ", "")  # Remove spaces for Chinese
        message = [
            {"role": "user", "content": f"{DEFAULT_SPEECH_TOKEN}请转写音频为文字"},
            {"role": "assistant", "content": text},
        ]
        messages.append(message)

    # Tokenize
    input_ids, attention_mask, target_ids = preprocess_texts(
        messages, tokenizer, max_len=128
    )
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    target_ids = target_ids.to(device)

    # Forward pass through encoder
    encoder_outs, enc_lengths, enc_mask = model.encoder(feature, feature_lengths)

    # Forward through projector
    speech_features, speech_lens = model.encoder_projector(encoder_outs, enc_lengths)

    # Get text embeddings
    inputs_embeds = model.llm.get_input_embeddings()(input_ids)

    # Merge speech and text embeddings
    inputs_embeds, attention_mask, target_ids = model._merge_input_ids_with_speech_features(
        speech_features.to(inputs_embeds.dtype),
        inputs_embeds,
        input_ids,
        attention_mask,
        target_ids,
        speech_lens=speech_lens
    )

    # Forward through LLM
    outputs = model.llm(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        labels=target_ids,
        return_dict=True
    )

    loss = outputs.loss

    # Track metrics
    info = MetricsTracker()
    info["loss"] = loss.detach().cpu()

    return loss, info


def compute_validation_loss(
    params: AttributeDict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: nn.Module,
    valid_dl: torch.utils.data.DataLoader,
    world_size: int = 1,
) -> MetricsTracker:
    """Compute validation loss"""
    model.eval()

    tot_loss = MetricsTracker()

    with torch.no_grad():
        for batch_idx, batch in enumerate(valid_dl):
            loss, info = compute_loss(
                params=params,
                tokenizer=tokenizer,
                model=model,
                batch=batch,
                is_training=False,
            )

            tot_loss["loss"] = info["loss"]

            if batch_idx >= 10:  # Limit validation batches
                break

    return tot_loss


def train_one_epoch(
    params: AttributeDict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    train_dl: torch.utils.data.DataLoader,
    valid_dl: torch.utils.data.DataLoader,
    tb_writer: Optional[SummaryWriter] = None,
    world_size: int = 1,
    rank: int = 0,
) -> None:
    """Train for one epoch"""

    # Set training mode
    model.train()
    model.encoder.eval()  # Encoder always in eval mode
    if params.training_stage == 1:
        model.llm.eval()  # LLM also in eval for Stage 1

    tot_loss = MetricsTracker()

    for batch_idx, batch in enumerate(train_dl):
        params.batch_idx_train += 1
        batch_size = len(batch["supervisions"]["text"])

        # Periodic validation
        if batch_idx % params.valid_interval == 0 and batch_idx > 0:
            logging.info("Running validation")
            valid_info = compute_validation_loss(
                params=params,
                tokenizer=tokenizer,
                model=model,
                valid_dl=valid_dl,
                world_size=world_size,
            )

            # Back to training mode
            model.train()
            model.encoder.eval()
            if params.training_stage == 1:
                model.llm.eval()

            # Enhanced validation logging
            valid_loss = valid_info.norm('loss')
            valid_msg = (
                f"[VALIDATION] Epoch {params.cur_epoch:2d} | "
                f"Loss: {valid_loss:.4f} | "
                f"Batches: 10"
            )

            # Use both print and logging for rank 0
            if rank == 0:
                print(valid_msg, flush=True)
            logging.info(valid_msg)
            sys.stdout.flush()

            if tb_writer is not None and rank == 0:
                tb_writer.add_scalar("valid/loss", valid_loss, params.batch_idx_train)
                tb_writer.flush()

            # Save checkpoint
            if rank == 0:
                save_checkpoint(params, model, optimizer, scheduler, train_dl.sampler)

        # Forward pass
        loss, info = compute_loss(
            params=params,
            tokenizer=tokenizer,
            model=model,
            batch=batch,
            is_training=True,
        )

        # Backward pass (DeepSpeed handles this)
        model.backward(loss)
        model.step()

        # Get current loss as scalar
        current_loss = loss.item()
        tot_loss["loss"] = current_loss

        # Logging - log first 5 batches regardless of log_interval for debugging
        should_log = (batch_idx % params.log_interval == 0) or (batch_idx < 5)

        if should_log:
            # Extract learning rate from DeepSpeed optimizer
            try:
                cur_lr = model.optimizer.param_groups[0]['lr']
            except:
                cur_lr = 0.0

            # Get gradient norm from DeepSpeed
            grad_norm = 0.0
            if hasattr(model, 'get_global_grad_norm'):
                try:
                    grad_norm = model.get_global_grad_norm()
                except:
                    pass

            # Get grad scale for FP16
            grad_scale = 1.0
            if hasattr(model, 'optimizer') and hasattr(model.optimizer, 'cur_scale'):
                try:
                    grad_scale = model.optimizer.cur_scale
                except:
                    pass

            # Console logging - detailed format
            log_msg = (
                f"[TRAIN] Epoch {params.cur_epoch:2d} | "
                f"Batch {batch_idx:4d} | "
                f"Loss {current_loss:.4f} | "
                f"Avg Loss {tot_loss.norm('loss'):.4f} | "
                f"LR {cur_lr:.2e} | "
                f"Grad Norm {grad_norm:.2f} | "
                f"Grad Scale {grad_scale:.1f} | "
                f"Batch Size {batch_size}"
            )

            # Use both logging and print for rank 0 to ensure output
            if rank == 0:
                print(log_msg, flush=True)
            logging.info(log_msg)
            sys.stdout.flush()  # Force flush for immediate output

            # TensorBoard logging (rank 0 only)
            if tb_writer is not None and rank == 0:
                step = params.batch_idx_train
                tb_writer.add_scalar("train/learning_rate", cur_lr, step)
                tb_writer.add_scalar("train/current_loss", current_loss, step)
                tb_writer.add_scalar("train/running_loss", tot_loss.norm("loss"), step)
                tb_writer.add_scalar("train/grad_norm", grad_norm, step)
                tb_writer.add_scalar("train/grad_scale", grad_scale, step)
                tb_writer.add_scalar("train/batch_size", batch_size, step)
                tb_writer.flush()  # Force flush TensorBoard

                # Reset metrics after writing
                tot_loss = MetricsTracker()


def save_checkpoint(
    params: AttributeDict,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    sampler,
) -> None:
    """Save checkpoint"""
    save_dir = Path(params.exp_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    tag = f"epoch-{params.cur_epoch}"

    # Save DeepSpeed checkpoint
    logging.info(f"Saving checkpoint to {save_dir}/{tag}")
    model.save_checkpoint(
        save_dir=str(save_dir),
        tag=tag,
        client_state={},
        exclude_frozen_parameters=True,
    )

    # Convert to FP32 state dict for easy loading
    logging.info("Converting checkpoint to FP32")
    convert_zero_checkpoint_to_fp32_state_dict(
        str(save_dir),
        f"{save_dir}/{tag}.pt",
        tag=tag,
        exclude_frozen_parameters=True,
    )

    # Save sampler state
    sampler_state_dict = sampler.state_dict()
    torch.save(sampler_state_dict, f"{save_dir}/{tag}-sampler.pt")

    logging.info(f"Checkpoint saved: {save_dir}/{tag}.pt")


def load_checkpoint(
    filename: Path,
    model: nn.Module,
) -> None:
    """Load checkpoint"""
    logging.info(f"Loading checkpoint from {filename}")
    checkpoint = torch.load(filename, map_location="cpu")
    model.load_state_dict(checkpoint, strict=False)
    logging.info("Checkpoint loaded successfully")


def estimate_training_steps(
    cuts,
    max_duration: float,
    num_epochs: int,
    world_size: int,
    gradient_accumulation_steps: int,
) -> int:
    """
    Estimate total training steps for learning rate scheduling.

    Args:
        cuts: Training CutSet (can be lazy)
        max_duration: Maximum duration per batch
        num_epochs: Number of training epochs
        world_size: Number of GPUs
        gradient_accumulation_steps: Gradient accumulation from DeepSpeed config

    Returns:
        Estimated total training steps
    """
    # Calculate total audio duration
    # This works with both lazy and materialized cuts
    total_duration = sum(cut.duration for cut in cuts)

    logging.info(f"Total training audio duration: {total_duration:.2f} seconds "
                 f"({total_duration/3600:.2f} hours)")

    # Estimate steps per epoch
    # Each batch can have up to max_duration seconds of audio
    batches_per_epoch = total_duration / max_duration

    # Account for distributed training
    steps_per_epoch = batches_per_epoch / world_size / gradient_accumulation_steps

    # Total steps across all epochs
    total_steps = int(steps_per_epoch * num_epochs)

    logging.info(f"Estimated training steps:")
    logging.info(f"  - Batches per epoch: {batches_per_epoch:.0f}")
    logging.info(f"  - Steps per epoch: {steps_per_epoch:.0f}")
    logging.info(f"  - Total steps ({num_epochs} epochs): {total_steps}")

    # Add 10% buffer to ensure decay completes
    total_steps_with_buffer = int(total_steps * 1.1)
    logging.info(f"  - Total steps (with 10% buffer): {total_steps_with_buffer}")

    return total_steps_with_buffer


def update_deepspeed_config_scheduler(
    config_path: str,
    total_num_steps: int,
    warmup_num_steps: int = 100,
) -> dict:
    """
    Load DeepSpeed config and update scheduler to use WarmupDecayLR.

    Args:
        config_path: Path to DeepSpeed JSON config
        total_num_steps: Total training steps (calculated)
        warmup_num_steps: Number of warmup steps

    Returns:
        Updated config dict
    """
    import json

    logging.info(f"Loading DeepSpeed config from: {config_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Get max LR from optimizer config
    max_lr = config.get('optimizer', {}).get('params', {}).get('lr', 1e-4)

    # Update scheduler to WarmupDecayLR
    old_scheduler = config.get('scheduler', {}).get('type', 'Unknown')
    logging.info(f"Replacing scheduler: {old_scheduler} -> WarmupDecayLR")

    config['scheduler'] = {
        "type": "WarmupDecayLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": max_lr,
            "warmup_num_steps": warmup_num_steps,
            "total_num_steps": total_num_steps
        }
    }

    logging.info(f"Scheduler configuration:")
    logging.info(f"  - Type: WarmupDecayLR")
    logging.info(f"  - Warmup: 0 -> {max_lr} over {warmup_num_steps} steps")
    logging.info(f"  - Total steps: {total_num_steps}")
    logging.info(f"  - Decay: Linear from step {warmup_num_steps} to {total_num_steps}")

    return config


def run(rank, world_size, args):
    """Main training function"""
    params = get_params()
    params.update(vars(args))

    # Setup logging directory
    params.exp_dir = Path(params.exp_dir)
    params.exp_dir.mkdir(parents=True, exist_ok=True)

    # CRITICAL: Set unbuffered output BEFORE any logging
    import sys
    import os

    # Force unbuffered output for Python
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
    sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)

    # Also set environment variable
    os.environ['PYTHONUNBUFFERED'] = '1'

    # Setup logger with explicit console handler
    setup_logger(
        f"{params.exp_dir}/log-train-{rank}",
        log_level="INFO",
        use_console=(rank == 0)  # Only rank 0 prints to console
    )

    # Test logging immediately - should appear on console for rank 0
    if rank == 0:
        print("=" * 80, flush=True)
        print(f"[INIT] Rank {rank}/{world_size} initialized successfully", flush=True)
        print(f"[INIT] Experiment directory: {params.exp_dir}", flush=True)
        print("=" * 80, flush=True)

    logging.info("=" * 80)
    logging.info(f"Rank {rank}/{world_size} initialized successfully")
    logging.info(f"Experiment directory: {params.exp_dir}")
    logging.info("=" * 80)
    sys.stdout.flush()

    logging.info("Training parameters:")
    logging.info(params)
    logging.info(f"World size: {world_size}, Rank: {rank}")
    sys.stdout.flush()

    # Set random seed
    fix_random_seed(params.seed + rank)

    # Build model
    model = build_model(params)

    # Get tokenizer
    tokenizer = LlmTokenizerWrapper.build_llm_tokenizer(params.llm_dir)

    # Validate chat template consistency
    if rank == 0:
        TRAINING_TEMPLATE = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content']}}{% if loop.last %}{{ '<|im_end|>'}}{% else %}{{ '<|im_end|>\n' }}{% endif %}{% endfor %}"
        validate_chat_template(TRAINING_TEMPLATE, stage="training")

    # Setup data BEFORE DeepSpeed initialization (needed for LR scheduler calculation)
    logging.info("Loading datasets")
    multi_dataset = MultiDataset(fbank_dir=str(params.manifest_dir))

    # Choose dataset based on parameters
    if params.use_test_dataset or params.dataset == "test":
        logging.info("Using synthetic test dataset")
        cuts_train = multi_dataset.test_dataset_train_cuts()
        cuts_valid = multi_dataset.test_dataset_dev_cuts()
    elif params.dataset == "aishell":
        logging.info("Using AISHELL-1 dataset")
        cuts_train = multi_dataset.aishell_train_cuts()
        cuts_valid = multi_dataset.aishell_dev_cuts()
    elif params.dataset == "all":
        logging.info("Using all Chinese ASR datasets")
        cuts_train = multi_dataset.train_cuts()
        cuts_valid = multi_dataset.dev_cuts()
    elif params.dataset == "custom":
        logging.info("Using custom dataset (urls.txt + ref.txt)")
        cuts_train = multi_dataset.custom_train_cuts()
        cuts_valid = multi_dataset.custom_dev_cuts()
    else:
        raise ValueError(f"Unknown dataset: {params.dataset}")

    sys.stdout.flush()

    # Create data loaders
    asr_datamodule = AsrDataModule(args)
    train_dl = asr_datamodule.train_dataloaders(cuts_train)
    valid_dl = asr_datamodule.valid_dataloaders(cuts_valid)

    # Initialize DeepSpeed with dynamic LR scheduler
    if params.deepspeed:
        # 1. Read original DeepSpeed config to get gradient_accumulation_steps
        import json
        import tempfile
        with open(params.deepspeed_config, 'r') as f:
            ds_config_original = json.load(f)
        gradient_accumulation_steps = ds_config_original.get('gradient_accumulation_steps', 1)

        # 2. Calculate total training steps
        logging.info("=" * 80)
        logging.info("Calculating learning rate schedule")
        total_num_steps = estimate_training_steps(
            cuts=cuts_train,
            max_duration=params.max_duration,
            num_epochs=params.num_epochs,
            world_size=world_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )

        # 3. Update DeepSpeed config with calculated steps
        ds_config_updated = update_deepspeed_config_scheduler(
            config_path=params.deepspeed_config,
            total_num_steps=total_num_steps,
            warmup_num_steps=100,  # Keep same warmup as before
        )

        # 4. Write updated config to temporary file
        # DeepSpeed requires config to be passed via args, not as config parameter
        temp_config_fd, temp_config_path = tempfile.mkstemp(suffix='.json', prefix='ds_config_', dir=params.exp_dir)
        with os.fdopen(temp_config_fd, 'w') as f:
            json.dump(ds_config_updated, f, indent=4)
        logging.info(f"Temporary DeepSpeed config written to: {temp_config_path}")

        # 5. Update args to point to temporary config
        original_config_path = params.deepspeed_config
        params.deepspeed_config = temp_config_path
        args.deepspeed_config = temp_config_path
        logging.info("=" * 80)

        # 6. Initialize DeepSpeed (will read config from args.deepspeed_config)
        logging.info("Initializing DeepSpeed with updated scheduler config")
        model, optimizer, _, scheduler = deepspeed.initialize(
            args=args,
            model=model,
            model_parameters=model.parameters()
        )

        # 7. Restore original config path in params (for reference)
        params.deepspeed_config = original_config_path
    else:
        raise ValueError("This script requires DeepSpeed. Please use --deepspeed flag.")

    # Load checkpoint if resuming training
    if params.start_epoch > 1:
        checkpoint_path = f"{params.exp_dir}/epoch-{params.start_epoch - 1}.pt"
        if os.path.exists(checkpoint_path):
            logging.info(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(checkpoint["model"], strict=False)
            params.batch_idx_train = checkpoint.get("batch_idx_train", 0)
            logging.info(f"Resumed training from epoch {params.start_epoch}")
            sys.stdout.flush()
        else:
            logging.warning(f"Checkpoint {checkpoint_path} not found, starting from epoch {params.start_epoch}")
            sys.stdout.flush()

    # Tensorboard
    tb_writer = None
    if params.tensorboard and rank == 0:
        tb_writer = SummaryWriter(log_dir=f"{params.exp_dir}/tensorboard")

    # Training loop
    for epoch in range(params.start_epoch, params.num_epochs + 1):
        params.cur_epoch = epoch
        fix_random_seed(params.seed + epoch)
        train_dl.sampler.set_epoch(epoch)

        # Print to ensure output visibility
        if rank == 0:
            print(f"\n{'='*80}", flush=True)
            print(f"[EPOCH] Starting epoch {epoch}/{params.num_epochs}", flush=True)
            print(f"{'='*80}\n", flush=True)

        logging.info(f"Starting epoch {epoch}")

        train_one_epoch(
            params=params,
            tokenizer=tokenizer,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_dl=train_dl,
            valid_dl=valid_dl,
            tb_writer=tb_writer,
            world_size=world_size,
            rank=rank,
        )

        # Save checkpoint at end of epoch
        if rank == 0:
            save_checkpoint(params, model, optimizer, scheduler, train_dl.sampler)

        logging.info(f"Epoch {epoch} completed")

    logging.info("Training completed!")


def main():
    parser = get_parser()
    args = parser.parse_args()

    world_size = get_world_size()
    rank = get_rank()

    run(rank, world_size, args)


if __name__ == "__main__":
    main()
