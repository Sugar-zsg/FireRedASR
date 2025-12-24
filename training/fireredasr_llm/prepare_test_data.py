#!/usr/bin/env python3
"""
Generate test dataset from a single audio file for testing FireRedASR-LLM training pipeline.

This script takes a single audio file and creates a synthetic dataset by:
1. Replicating it multiple times with different utterance IDs
2. Extracting 80-dim Fbank features using kaldi_native_fbank
3. Creating fake Chinese transcriptions
4. Generating Lhotse CutSet manifests (train/dev/test splits)

Usage:
    python prepare_test_data.py \
        --audio-path /path/to/audio.wav \
        --output-dir data/fbank/test_dataset \
        --num-samples 100
"""

import argparse
import logging
import shutil
from pathlib import Path
from typing import List

import kaldi_native_fbank as knf
import numpy as np
import soundfile as sf
from lhotse import AudioSource, CutSet, Recording, SupervisionSegment, RecordingSet, SupervisionSet
from lhotse.cut import MonoCut
from lhotse.features import Fbank, FbankConfig


def get_parser():
    parser = argparse.ArgumentParser(
        description="Generate test dataset for FireRedASR-LLM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--audio-path",
        type=str,
        default="/workspace/bella-infra/user/zhangshuge002/dataset_test/output_0_10_1742561773196.wav",
        help="Path to source audio file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/fbank/test_dataset"),
        help="Output directory for manifests and features",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Total number of samples to generate",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Ratio of training samples (rest split between dev and test)",
    )
    parser.add_argument(
        "--num-mel-bins",
        type=int,
        default=80,
        help="Number of mel filterbanks (must be 80 for FireRedASR)",
    )
    return parser


def extract_fbank_features(audio_path: str, num_mel_bins: int = 80) -> np.ndarray:
    """
    Extract Fbank features from audio file using kaldi_native_fbank.

    Args:
        audio_path: Path to audio file
        num_mel_bins: Number of mel filterbanks

    Returns:
        features: (num_frames, num_mel_bins) numpy array
    """
    # Load audio
    audio, sample_rate = sf.read(audio_path)

    # Ensure mono
    if len(audio.shape) > 1:
        audio = audio[:, 0]

    # Convert to float32 if needed
    audio = audio.astype(np.float32)

    # Create Fbank options
    opts = knf.FbankOptions()
    opts.frame_opts.dither = 0.0
    opts.frame_opts.snip_edges = False
    opts.frame_opts.samp_freq = sample_rate
    opts.mel_opts.num_bins = num_mel_bins
    opts.frame_opts.frame_shift_ms = 10.0  # 10ms shift
    opts.frame_opts.frame_length_ms = 25.0  # 25ms window

    # Extract features
    fbank = knf.OnlineFbank(opts)
    fbank.accept_waveform(sample_rate, audio.tolist())
    fbank.input_finished()

    # Get features
    num_frames = fbank.num_frames_ready
    features = []
    for i in range(num_frames):
        frame = fbank.get_frame(i)
        features.append(frame)

    return np.array(features, dtype=np.float32)


def generate_fake_transcription(idx: int) -> str:
    """Generate fake Chinese transcription for testing."""
    # Variety of Chinese test sentences
    templates = [
        "这是第{}个测试音频样本",
        "火红语音识别系统测试数据编号{}",
        "人工智能语音转写测试样本{}",
        "深度学习模型训练数据{}",
        "自动语音识别评估集{}号",
        "语音信号处理实验数据{}",
        "端到端语音识别测试{}",
        "多模态大模型训练样本{}",
        "声学特征提取验证数据{}",
        "神经网络模型评估样本{}",
    ]
    template = templates[idx % len(templates)]
    return template.format(idx)


def create_test_dataset(args):
    """Main function to create test dataset."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    )

    # Create output directories
    args.output_dir.mkdir(parents=True, exist_ok=True)
    feats_dir = args.output_dir / "feats"
    feats_dir.mkdir(exist_ok=True)

    logging.info(f"Loading source audio: {args.audio_path}")

    # Load source audio info
    audio_info = sf.info(args.audio_path)
    duration = audio_info.duration
    sample_rate = audio_info.samplerate

    logging.info(f"Audio duration: {duration:.2f}s, sample rate: {sample_rate}Hz")

    # Create recordings and supervisions
    logging.info(f"Generating {args.num_samples} synthetic samples...")
    recordings = []
    supervisions = []

    for i in range(args.num_samples):
        utt_id = f"test_utt_{i:05d}"

        # Create recording
        recording = Recording(
            id=utt_id,
            sources=[
                AudioSource(
                    type="file",
                    channels=[0],
                    source=str(args.audio_path),
                )
            ],
            sampling_rate=sample_rate,
            num_samples=int(duration * sample_rate),
            duration=duration,
        )
        recordings.append(recording)

        # Generate transcription
        text = generate_fake_transcription(i)

        # Create supervision
        supervision = SupervisionSegment(
            id=utt_id,
            recording_id=utt_id,
            start=0,
            duration=duration,
            channel=0,
            text=text,
            language="Chinese",
        )
        supervisions.append(supervision)

        if (i + 1) % 20 == 0:
            logging.info(f"Generated {i + 1}/{args.num_samples} samples")

    # Create RecordingSet and SupervisionSet
    recording_set = RecordingSet.from_recordings(recordings)
    supervision_set = SupervisionSet.from_segments(supervisions)

    # Create CutSet from manifests
    cuts = CutSet.from_manifests(recordings=recording_set, supervisions=supervision_set)
    logging.info(f"Created CutSet with {len(cuts)} cuts")

    # Extract and store features using Lhotse's standard method
    logging.info(f"Extracting and storing features...")
    fbank_extractor = Fbank(FbankConfig(num_mel_bins=args.num_mel_bins))
    cuts = cuts.compute_and_store_features(
        extractor=fbank_extractor,
        storage_path=str(feats_dir),
        num_jobs=1,  # Single job for test data
    )
    logging.info(f"Features extracted and stored")

    # Split into train/dev/test
    num_train = int(args.num_samples * args.train_ratio)
    num_dev = int(args.num_samples * (1 - args.train_ratio) / 2)
    num_test = args.num_samples - num_train - num_dev

    cuts_train = cuts.subset(first=num_train)
    remaining = cuts.subset(last=args.num_samples - num_train)
    cuts_dev = remaining.subset(first=num_dev)
    cuts_test = remaining.subset(last=num_test)

    logging.info(f"Split: train={len(cuts_train)}, dev={len(cuts_dev)}, test={len(cuts_test)}")

    # Save manifests
    train_path = args.output_dir / "test_cuts_train.jsonl.gz"
    dev_path = args.output_dir / "test_cuts_dev.jsonl.gz"
    test_path = args.output_dir / "test_cuts_test.jsonl.gz"

    logging.info(f"Saving train manifest to {train_path}")
    cuts_train.to_file(train_path)

    logging.info(f"Saving dev manifest to {dev_path}")
    cuts_dev.to_file(dev_path)

    logging.info(f"Saving test manifest to {test_path}")
    cuts_test.to_file(test_path)

    # Print summary
    logging.info("\n" + "="*60)
    logging.info("Test dataset generation completed!")
    logging.info("="*60)
    logging.info(f"Output directory: {args.output_dir}")
    logging.info(f"Train samples: {len(cuts_train)} ({train_path})")
    logging.info(f"Dev samples: {len(cuts_dev)} ({dev_path})")
    logging.info(f"Test samples: {len(cuts_test)} ({test_path})")
    logging.info(f"Feature dimensions: {args.num_mel_bins}")
    logging.info(f"Feature directory: {feats_dir}")
    logging.info("")
    logging.info("Example transcriptions:")
    for i, cut in enumerate(cuts_train):
        if i >= 5:
            break
        logging.info(f"  {cut.id}: {cut.supervisions[0].text}")
    logging.info("")
    logging.info("To use this dataset for training:")
    logging.info(f"  python training/fireredasr_llm/train.py \\")
    logging.info(f"    --manifest-dir {args.output_dir} \\")
    logging.info(f"    --training-stage 1 \\")
    logging.info(f"    --encoder-path pretrained_models/FireRedASR-AED-L/model.pth.tar \\")
    logging.info(f"    --llm-dir pretrained_models/Qwen2-7B-Instruct \\")
    logging.info(f"    --exp-dir exp/test_run \\")
    logging.info(f"    --num-epochs 2 \\")
    logging.info(f"    --max-duration 50 \\")
    logging.info(f"    --deepspeed \\")
    logging.info(f"    --deepspeed_config training/fireredasr_llm/ds_config_stage1.json")
    logging.info("="*60 + "\n")


def main():
    parser = get_parser()
    args = parser.parse_args()
    create_test_dataset(args)


if __name__ == "__main__":
    main()
