#!/usr/bin/env python3

import argparse
import glob
import os
import sys

# Ensure HF registration for FireRedASR
import fireredasr.hf  # noqa: F401
from transformers import AutoModel
from fireredasr.hf import FireRedASRConfig


parser = argparse.ArgumentParser()
parser.add_argument('--asr_type', type=str, required=True, choices=["aed", "llm"])
parser.add_argument('--model_dir', type=str, required=True)

# Input / Output
parser.add_argument("--wav_path", type=str)
parser.add_argument("--wav_paths", type=str, nargs="*")
parser.add_argument("--wav_dir", type=str)
parser.add_argument("--wav_scp", type=str)
parser.add_argument("--output", type=str)

# Decode Options
parser.add_argument('--use_gpu', type=int, default=1)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--beam_size", type=int, default=1)
parser.add_argument("--decode_max_len", type=int, default=0)
# FireRedASR-AED
parser.add_argument("--nbest", type=int, default=1)
parser.add_argument("--softmax_smoothing", type=float, default=1.0)
parser.add_argument("--aed_length_penalty", type=float, default=0.0)
parser.add_argument("--eos_penalty", type=float, default=1.0)
# FireRedASR-LLM
parser.add_argument("--decode_min_len", type=int, default=0)
parser.add_argument("--repetition_penalty", type=float, default=1.0)
parser.add_argument("--llm_length_penalty", type=float, default=0.0)
parser.add_argument("--temperature", type=float, default=1.0)


def main(args):
    wavs = get_wav_info(args)
    fout = open(args.output, "w") if args.output else None

    # Load via Transformers AutoModel with our custom config
    config = FireRedASRConfig(asr_type=args.asr_type)
    model = AutoModel.from_pretrained(
        args.model_dir,
        config=config,
        use_gpu=args.use_gpu,
    )

    batch_uttid = []
    batch_wav_path = []
    for i, wav in enumerate(wavs):
        uttid, wav_path = wav
        batch_uttid.append(uttid)
        batch_wav_path.append(wav_path)
        if len(batch_wav_path) < args.batch_size and i != len(wavs) - 1:
            continue

        # Hugging Face-style generation
        out = model.generate(
            wav_paths=batch_wav_path,
            beam_size=args.beam_size,
            max_new_tokens=args.decode_max_len,
            nbest=args.nbest,
            repetition_penalty=args.repetition_penalty,
            length_penalty=(
                args.aed_length_penalty if args.asr_type == "aed" else args.llm_length_penalty
            ),
            eos_penalty=args.eos_penalty,
            temperature=args.temperature,
        )

        for uttid, text in zip(batch_uttid, out.sequences):
            result = {"uttid": uttid, "text": text, "rtf": f"{out.rtf:.4f}"}
            print(result)
            if fout is not None:
                fout.write(f"{uttid}\t{text}\n")

        batch_uttid = []
        batch_wav_path = []


def get_wav_info(args):
    """
    Returns:
        wavs: list of (uttid, wav_path)
    """
    base = lambda p: os.path.basename(p).replace(".wav", "")
    if args.wav_path:
        wavs = [(base(args.wav_path), args.wav_path)]
    elif args.wav_paths and len(args.wav_paths) >= 1:
        wavs = [(base(p), p) for p in sorted(args.wav_paths)]
    elif args.wav_scp:
        wavs = [line.strip().split() for line in open(args.wav_scp)]
    elif args.wav_dir:
        wavs = glob.glob(f"{args.wav_dir}/**/*.wav", recursive=True)
        wavs = [(base(p), p) for p in sorted(wavs)]
    else:
        raise ValueError("Please provide valid wav info")
    print(f"#wavs={len(wavs)}")
    return wavs


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    main(args)
