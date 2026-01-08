#!/usr/bin/env python3
import argparse
import os

from transformers import AutoModel

from fireredasr.hf import FireRedASRConfig  # ensures Auto registration


def main():
    parser = argparse.ArgumentParser(
        description="Package FireRedASR weights for Transformers AutoModel usage."
    )
    parser.add_argument("--src_dir", required=True, help="Source checkpoint dir")
    parser.add_argument("--dst_dir", required=True, help="Destination output dir")
    parser.add_argument(
        "--asr_type",
        required=True,
        choices=["aed", "llm"],
        help="Model type to export",
    )
    parser.add_argument(
        "--llm_repo_id",
        default=None,
        help="Optional HF repo id for the LLM (avoids copying LLM folder)",
    )
    parser.add_argument(
        "--copy_llm_dir",
        action="store_true",
        help="Copy the LLM directory into the package (can be very large)",
    )
    parser.add_argument(
        "--include_code",
        action="store_true",
        help="Include minimal runtime code to enable trust_remote_code loading",
    )
    args = parser.parse_args()

    config = FireRedASRConfig(asr_type=args.asr_type)
    model = AutoModel.from_pretrained(args.src_dir, config=config, use_gpu=False)

    save_kwargs = {"src_dir": args.src_dir}
    if args.llm_repo_id:
        save_kwargs["llm_repo_id"] = args.llm_repo_id
    if args.copy_llm_dir:
        save_kwargs["copy_llm_dir"] = True

    os.makedirs(args.dst_dir, exist_ok=True)
    if args.include_code:
        save_kwargs["include_code"] = True
    model.save_pretrained(args.dst_dir, **save_kwargs)

    # Write a lightweight README for the packaged model
    readme = os.path.join(args.dst_dir, "README.md")
    with open(readme, "w", encoding="utf-8") as f:
        f.write(
            """# FireRedASR (Transformers Package)\n\n"
            "This folder contains packaged FireRedASR artifacts for inference via Transformers.\n\n"
            "## Usage\n"
            "```python\n"
            "from transformers import AutoModel\n"
            "from fireredasr.hf import FireRedASRConfig  # pip install fireredasr\n\n"
            "config = FireRedASRConfig(asr_type=\"aed\")  # or 'llm'\n"
            "model = AutoModel.from_pretrained('.', config=config, trust_remote_code=True)\n"
            "out = model.generate(wav_paths=['examples/wav/BAC009S0764W0121.wav'])\n"
            "print(out.sequences, out.rtf)\n"
            "```\n\n"
            "## Files\n"
            "- config.json (HF config)\n"
            "- model.pth.tar, cmvn.ark (+ dict.txt, train_bpe1000.model for AED)\n"
            "- asr_encoder.pth.tar for LLM; LLM weights via llm_dir or llm_repo_id.\n"
        )

    print(f"Exported to: {args.dst_dir}")


if __name__ == "__main__":
    main()
