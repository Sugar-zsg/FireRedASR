#!/usr/bin/env python3
# Evaluation script for FireRedASR-LLM

import argparse
import logging
from pathlib import Path
from typing import Dict, List

import torch
from lhotse import load_manifest_lazy
from lhotse.dataset import DynamicBucketingSampler, K2SpeechRecognitionDataset, PrecomputedFeatures
from torch.utils.data import DataLoader

# Import from FireRedASR
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from fireredasr.models.fireredasr_llm import FireRedAsrLlm
from fireredasr.tokenizer.llm_tokenizer import LlmTokenizerWrapper

# Import utilities
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.dist import get_rank, get_world_size


def get_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate FireRedASR-LLM on test sets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained checkpoint (.pt file)",
    )
    parser.add_argument(
        "--encoder-path",
        type=str,
        required=True,
        help="Path to FireRedASR-AED checkpoint",
    )
    parser.add_argument(
        "--llm-dir",
        type=str,
        required=True,
        help="Path to Qwen2 model directory",
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path("data/fbank"),
        help="Path to manifest directory",
    )
    parser.add_argument(
        "--test-manifest",
        type=str,
        default="test_cuts.jsonl.gz",
        help="Test manifest filename (e.g., aishell_cuts_test.jsonl.gz)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory to save results",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=100.0,
        help="Max duration per batch (seconds)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of dataloader workers",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=3,
        help="Beam size for decoding",
    )
    parser.add_argument(
        "--decode-max-len",
        type=int,
        default=0,
        help="Max decode length (0=auto)",
    )
    parser.add_argument(
        "--decode-min-len",
        type=int,
        default=0,
        help="Min decode length",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=3.0,
        help="Repetition penalty",
    )
    parser.add_argument(
        "--llm-length-penalty",
        type=float,
        default=1.0,
        help="Length penalty",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--use-gpu",
        type=int,
        default=1,
        help="Use GPU for inference (0 or 1)",
    )

    return parser


def load_model(args):
    """Load trained model from checkpoint"""
    logging.info(f"Loading model from {args.checkpoint}")

    # Build model
    model_args = argparse.Namespace(
        encoder_path=args.encoder_path,
        llm_dir=args.llm_dir,
        freeze_encoder=True,
        freeze_llm=False,
        use_lora=True,  # Assume Stage 2 checkpoint
        use_fp16=False,  # Use FP32 for inference
        use_flash_attn=False,
        encoder_downsample_rate=2
    )
    model = FireRedAsrLlm.from_args(model_args)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    missing, unexpected = model.load_state_dict(checkpoint, strict=False)
    logging.info(f"Checkpoint loaded. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

    model.eval()

    # Move to GPU if available
    if args.use_gpu and torch.cuda.is_available():
        model = model.cuda()
        logging.info("Model moved to GPU")

    return model


def load_test_data(args):
    """Load test data from manifest"""
    manifest_path = args.manifest_dir / args.test_manifest
    logging.info(f"Loading test data from {manifest_path}")

    cuts = load_manifest_lazy(manifest_path)
    logging.info(f"Loaded {len(cuts)} test utterances")

    # Create dataset
    dataset = K2SpeechRecognitionDataset(
        input_strategy=PrecomputedFeatures(),
        return_cuts=True,
    )

    # Create sampler
    sampler = DynamicBucketingSampler(
        cuts,
        max_duration=args.max_duration,
        shuffle=False,
    )

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=None,
        sampler=sampler,
        num_workers=args.num_workers,
    )

    return dataloader


def decode_batch(model, tokenizer, batch, args):
    """
    Decode a batch using model.

    Returns:
        List of (uttid, hypothesis_text) tuples
    """
    device = next(model.parameters()).device

    # Get features
    features = batch["inputs"]  # (N, T, F)
    features = features.to(device)
    feature_lengths = batch["supervisions"]["num_frames"].to(device)

    # Get utterance IDs
    uttids = batch["supervisions"]["cut"].ids if "cut" in batch["supervisions"] else \
             [f"utt_{i}" for i in range(len(batch["supervisions"]["text"]))]

    # Encode features
    encoder_outs, enc_lengths, enc_mask = model.encoder(features, feature_lengths)

    # Project to LLM space
    speech_features, speech_lens = model.encoder_projector(encoder_outs, enc_lengths)

    # Prepare initial tokens for decoding (user prompt with <speech>)
    from fireredasr.tokenizer.llm_tokenizer import DEFAULT_SPEECH_TOKEN
    messages = []
    for _ in range(speech_features.size(0)):
        message = [
            {"role": "user", "content": f"{DEFAULT_SPEECH_TOKEN}请转写音频为文字"},
        ]
        messages.append(message)

    # Tokenize without assistant response
    TEMPLATE = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n'}}{% endfor %}"
    texts = []
    for msg in messages:
        texts.append(
            tokenizer.apply_chat_template(
                msg,
                tokenize=True,
                chat_template=TEMPLATE,
                add_generation_prompt=True,  # Add assistant prompt
                padding="longest",
                max_length=128,
                truncation=True,
            )
        )

    # Pad texts
    max_len_texts = max([len(text) for text in texts])
    if tokenizer.padding_side == "right":
        texts = [text + [tokenizer.pad_token_id] * (max_len_texts - len(text)) for text in texts]
    else:
        texts = [[tokenizer.pad_token_id] * (max_len_texts - len(text)) + text for text in texts]

    input_ids = torch.tensor(texts, dtype=torch.long).to(device)
    attention_mask = input_ids.ne(tokenizer.pad_token_id).to(device)

    # Get embeddings
    inputs_embeds = model.llm.get_input_embeddings()(input_ids)

    # Merge with speech features
    inputs_embeds, attention_mask, _ = model._merge_input_ids_with_speech_features(
        speech_features.to(inputs_embeds.dtype),
        inputs_embeds,
        input_ids,
        attention_mask,
        None,  # No target_ids for inference
        speech_lens=speech_lens
    )

    # Generate
    with torch.no_grad():
        generated_ids = model.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=args.decode_max_len if args.decode_max_len > 0 else 128,
            min_new_tokens=args.decode_min_len,
            num_beams=args.beam_size,
            repetition_penalty=args.repetition_penalty,
            length_penalty=args.llm_length_penalty,
            temperature=args.temperature,
            do_sample=False if args.beam_size > 1 else True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode generated text
    hypotheses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    # Clean up (remove prompt if present)
    results = []
    for uttid, hyp in zip(uttids, hypotheses):
        # Extract only the response part (after the prompt)
        hyp = hyp.strip()
        if "请转写音频为文字" in hyp:
            hyp = hyp.split("请转写音频为文字")[-1].strip()
        results.append((uttid, hyp))

    return results


def compute_wer(refs: List[str], hyps: List[str]) -> Dict[str, float]:
    """
    Compute WER/CER statistics.

    Returns:
        Dict with wer, cer, insertions, deletions, substitutions
    """
    try:
        import kaldialign
    except ImportError:
        logging.warning("kaldialign not installed, cannot compute WER. Install with: pip install kaldialign")
        return {"wer": 0.0, "cer": 0.0}

    # Compute character-level error rate (for Chinese)
    total_chars = 0
    total_errors = 0

    for ref, hyp in zip(refs, hyps):
        # Remove spaces for Chinese
        ref_chars = list(ref.replace(" ", ""))
        hyp_chars = list(hyp.replace(" ", ""))

        total_chars += len(ref_chars)

        # Align and count errors
        ali = kaldialign.align(ref_chars, hyp_chars, "*")
        errors = sum(1 for r, h in ali if r != h)
        total_errors += errors

    cer = 100.0 * total_errors / total_chars if total_chars > 0 else 0.0

    return {
        "cer": cer,
        "total_chars": total_chars,
        "total_errors": total_errors,
    }


def evaluate(args):
    """Main evaluation function"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    )

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = load_model(args)
    tokenizer = LlmTokenizerWrapper.build_llm_tokenizer(args.llm_dir)

    # Load test data
    test_dl = load_test_data(args)

    # Run inference
    logging.info("Starting evaluation...")
    all_results = []
    all_refs = []
    all_hyps = []

    for batch_idx, batch in enumerate(test_dl):
        if batch_idx % 10 == 0:
            logging.info(f"Processing batch {batch_idx}...")

        # Get references
        refs = batch["supervisions"]["text"]

        # Decode
        results = decode_batch(model, tokenizer, batch, args)

        # Collect results
        for (uttid, hyp), ref in zip(results, refs):
            all_results.append((uttid, ref, hyp))
            all_refs.append(ref)
            all_hyps.append(hyp)

            # Print first few examples
            if len(all_results) <= 5:
                logging.info(f"Example - REF: {ref}")
                logging.info(f"Example - HYP: {hyp}")
                logging.info("")

    # Compute metrics
    logging.info("\n" + "="*60)
    logging.info("Computing metrics...")
    metrics = compute_wer(all_refs, all_hyps)

    logging.info(f"Results on {len(all_results)} utterances:")
    logging.info(f"CER: {metrics['cer']:.2f}%")
    if 'total_errors' in metrics:
        logging.info(f"Total errors: {metrics['total_errors']} / {metrics['total_chars']} chars")
    logging.info("="*60 + "\n")

    # Save results
    output_file = args.output_dir / f"{Path(args.test_manifest).stem}_results.txt"
    logging.info(f"Saving results to {output_file}")

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"CER: {metrics['cer']:.2f}%\n")
        f.write(f"Total utterances: {len(all_results)}\n")
        f.write("="*60 + "\n\n")

        for uttid, ref, hyp in all_results:
            f.write(f"ID: {uttid}\n")
            f.write(f"REF: {ref}\n")
            f.write(f"HYP: {hyp}\n")
            f.write("\n")

    logging.info(f"Evaluation completed! Results saved to {output_file}")


def main():
    parser = get_parser()
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
