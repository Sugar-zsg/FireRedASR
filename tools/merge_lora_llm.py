#!/usr/bin/env python3
"""
Merge LoRA adapters into the base LLM and save a standalone LLM folder.

Usage (run under repo root or FireRedASR/):

  cd FireRedASR
  python tools/merge_lora_llm.py \
    --model_dir pretrained_models/FireRedASR-LLM-L \
    --save_dir  pretrained_models/FireRedASR-LLM-L-Merged

After running, you can infer with:

  speech2text.py --asr_type llm --model_dir pretrained_models/FireRedASR-LLM-L-Merged ...

This script:
  - Loads FireRedASR-LLM model and LoRA weights from model.pth.tar
  - Merges LoRA into the base LLM (Qwen2-7B-Instruct)
  - Saves merged LLM to <save_dir>/Qwen2-7B-Instruct
  - Copies cmvn.ark and asr_encoder.pth.tar to <save_dir>
  - Writes an updated model.pth.tar with args.use_lora=False
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import torch


WEIGHT_SUFFIXES = (".safetensors", ".bin")


def _has_weight_files(model_dir: Path, min_size_mb: int = 10) -> bool:
    """Rudimentary check for presence of large weight files.

    Returns True if any *.safetensors or *.bin exceeds min_size_mb.
    """
    for p in model_dir.glob("**/*"):
        if p.is_file() and p.suffix in WEIGHT_SUFFIXES:
            try:
                if p.stat().st_size >= min_size_mb * 1024 * 1024:
                    return True
            except OSError:
                continue
    return False


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", type=str, required=True,
                   help="Source FireRedASR-LLM model dir (contains model.pth.tar, asr_encoder.pth.tar, cmvn.ark, Qwen2-7B-Instruct)")
    p.add_argument("--save_dir", type=str, required=True,
                   help="Destination dir to save merged model")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"],
                   help="Device to build model for merging; 'auto' prefers CUDA if available")
    p.add_argument("--force", action="store_true", help="Overwrite save_dir if exists")
    return p.parse_args()


def main():
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    # Ensure imports work both from repo root and nested
    sys.path.insert(0, str(repo_root))

    model_dir = (repo_root / args.model_dir).resolve()
    save_dir = (repo_root / args.save_dir).resolve()
    llm_dir = model_dir / "Qwen2-7B-Instruct"
    model_tar = model_dir / "model.pth.tar"
    enc_tar = model_dir / "asr_encoder.pth.tar"
    cmvn_ark = model_dir / "cmvn.ark"

    if not model_tar.exists():
        raise FileNotFoundError(f"Missing model.pth.tar: {model_tar}")
    if not enc_tar.exists():
        raise FileNotFoundError(f"Missing asr_encoder.pth.tar: {enc_tar}")
    if not cmvn_ark.exists():
        raise FileNotFoundError(f"Missing cmvn.ark: {cmvn_ark}")
    if not llm_dir.exists():
        raise FileNotFoundError(f"Missing base LLM dir: {llm_dir}")
    if not _has_weight_files(llm_dir):
        raise FileNotFoundError(
            f"No large weight files found under {llm_dir}.\n"
            f"Please download Qwen2-7B-Instruct weights locally, e.g.:\n"
            f"  huggingface-cli download Qwen/Qwen2-7B-Instruct \\n+    --local-dir {llm_dir} --local-dir-use-symlinks False\n"
            f"Then re-run this script."
        )

    if save_dir.exists():
        if not args.force:
            raise FileExistsError(f"save_dir already exists: {save_dir}. Use --force to overwrite.")
        # Clean target dir if forcing
        shutil.rmtree(save_dir)
    (save_dir / "Qwen2-7B-Instruct").mkdir(parents=True, exist_ok=True)

    # Deferred import after sys.path adjustment
    from fireredasr.models.fireredasr_llm import FireRedAsrLlm
    from transformers import AutoTokenizer

    # Decide device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda" if (args.device == "cuda") else "cpu")

    print(f"[Merge] Loading package: {model_tar}")
    pkg = torch.load(str(model_tar), map_location="cpu")
    train_args = pkg["args"]

    # Ensure paths
    train_args.encoder_path = str(enc_tar)
    train_args.llm_dir = str(llm_dir)
    # Make sure we build with LoRA enabled to get a PEFT-wrapped LLM
    # (Will be a no-op if the checkpoint wasn't trained with LoRA.)
    train_args.freeze_llm = False
    train_args.use_lora = True

    print("[Merge] Building FireRedAsrLlm (with PEFT if available)...")
    model = FireRedAsrLlm.from_args(train_args)
    missing, unexpected = model.load_state_dict(pkg.get("model_state_dict",{}), strict=False)
    if missing:
        print(f"[Merge] Warning: {len(missing)} missing keys (expected for base LLM weights)")
    if unexpected:
        print(f"[Merge] Info: {len(unexpected)} unexpected keys (often LoRA adapter keys)")
    model = model.to(device)
    llm = model.llm

    # Try to merge LoRA if wrapped by PEFT
    merged = False
    # Prefer explicit PEFT type check when available
    try:
        from peft import PeftModel  # type: ignore
    except Exception:
        PeftModel = None  # type: ignore

    if PeftModel is not None and isinstance(llm, PeftModel):
        print("[Merge] Detected PEFT-wrapped LLM; merging adapters...")
        llm = llm.merge_and_unload()
        merged = True
    elif hasattr(llm, "merge_and_unload"):
        print("[Merge] Merging LoRA into base LLM via merge_and_unload() ...")
        llm = llm.merge_and_unload()
        merged = True
    else:
        print("[Merge] No LoRA detected or PEFT not available; proceeding without merge.")

    if merged:
        print("[Merge] LoRA successfully merged into base LLM.")
    else:
        print("[Merge] No LoRA detected or merge not supported; saving current LLM as-is.")

    # Save merged LLM and tokenizer
    save_llm_dir = save_dir / "Qwen2-7B-Instruct"
    print(f"[Merge] Saving merged LLM to: {save_llm_dir}")
    # Move to CPU before saving to ensure full weights are written
    llm = llm.to("cpu")
    llm.save_pretrained(str(save_llm_dir), safe_serialization=True, max_shard_size="2GB")

    # Sanity check: ensure we actually wrote substantial weights
    if not _has_weight_files(save_llm_dir):
        raise RuntimeError(
            f"Saved LLM at {save_llm_dir} lacks weight shards.\n"
            f"This typically means the base LLM wasn't loaded. Ensure {llm_dir} contains full weights and re-run."
        )
    AutoTokenizer.from_pretrained(str(llm_dir)).save_pretrained(str(save_llm_dir))

    # Copy auxiliary files
    for src in [enc_tar, cmvn_ark]:
        dst = save_dir / src.name
        print(f"[Merge] Copying {src.name} -> {dst}")
        shutil.copy2(src, dst)

    # Update and save package: disable LoRA for future loads
    pkg["args"].llm_dir = str(save_llm_dir)
    pkg["args"].use_lora = False
    save_tar = save_dir / "model.pth.tar"
    print(f"[Merge] Writing updated package (use_lora=False): {save_tar}")
    torch.save(pkg, str(save_tar))

    print("[Merge] Done. You can now use:")
    print(f"  speech2text.py --asr_type llm --model_dir {save_dir} ...")


if __name__ == "__main__":
    main()
