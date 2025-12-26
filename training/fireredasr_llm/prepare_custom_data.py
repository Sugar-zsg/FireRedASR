#!/usr/bin/env python3
"""
从urls.txt和ref.txt生成Lhotse格式的训练数据集
"""
import argparse
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple

import torch
import torchaudio
from lhotse import CutSet, Recording, SupervisionSegment, MonoCut
from lhotse import Fbank, FbankConfig
from lhotse.audio import AudioSource
from lhotse.features.io import LilcomFilesWriter
from tqdm import tqdm
import urllib.request
import hashlib


def setup_logger():
    """设置日志"""
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def download_audio(url: str, output_path: str, timeout: int = 30) -> bool:
    """下载音频文件，支持超时设置"""
    try:
        logging.info(f"Downloading: {url}")

        # 使用socket设置全局超时
        import socket
        socket.setdefaulttimeout(timeout)

        # 下载文件
        urllib.request.urlretrieve(url, output_path)

        # 恢复默认超时
        socket.setdefaulttimeout(None)
        return True
    except socket.timeout:
        logging.warning(f"Download timeout ({timeout}s) for {url}, skipping...")
        return False
    except Exception as e:
        logging.warning(f"Failed to download {url}: {e}, skipping...")
        return False


def convert_audio_to_16k(input_path: str, output_path: str) -> bool:
    """将音频转换为16kHz单声道格式"""
    try:
        # 使用ffmpeg转换为16kHz 16-bit PCM mono WAV
        cmd = [
            'ffmpeg', '-y', '-i', input_path,
            '-ar', '16000',
            '-ac', '1',
            '-acodec', 'pcm_s16le',
            '-f', 'wav',
            output_path
        ]
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30
        )
        if result.returncode == 0:
            return True
        else:
            logging.error(f"FFmpeg error: {result.stderr.decode()}")
            return False
    except Exception as e:
        logging.error(f"Failed to convert {input_path}: {e}")
        return False


def get_audio_duration(audio_path: str) -> float:
    """获取音频时长（秒）"""
    try:
        # Use soundfile (more reliable) or fallback to ffprobe
        import soundfile as sf
        info = sf.info(audio_path)
        return info.duration
    except:
        # Fallback: use ffprobe
        try:
            import subprocess
            result = subprocess.run(
                ['ffprobe', '-v', 'error', '-show_entries',
                 'format=duration', '-of',
                 'default=noprint_wrappers=1:nokey=1', audio_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=10
            )
            return float(result.stdout)
        except Exception as e:
            logging.error(f"Failed to get duration for {audio_path}: {e}")
            return 0.0


def create_cuts_from_urls(
    urls: List[str],
    texts: List[str],
    audio_dir: Path,
    max_duration: float = 30.0,
    min_duration: float = 0.5,
    download_timeout: int = 30,
) -> List[MonoCut]:
    """从URLs创建Lhotse CutSet

    Args:
        urls: 音频URL或本地路径列表
        texts: 对应的参考文本列表
        audio_dir: 音频保存目录
        max_duration: 最大音频时长（秒）
        min_duration: 最小音频时长（秒）
        download_timeout: 下载超时时间（秒），超过此时间会跳过该数据
    """

    cuts = []
    audio_dir.mkdir(parents=True, exist_ok=True)

    skipped_count = 0

    for idx, (url, text) in enumerate(tqdm(zip(urls, texts), total=len(urls), desc="Processing audios")):
        # 生成唯一ID（基于URL hash）
        url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
        utt_id = f"custom_{idx:05d}_{url_hash}"

        # 下载音频（带超时控制）
        temp_audio = audio_dir / f"{utt_id}_temp.wav"
        final_audio = audio_dir / f"{utt_id}.wav"

        if not download_audio(url, str(temp_audio), timeout=download_timeout):
            logging.warning(f"Skipping {utt_id} and its corresponding text (download failed or timeout)")
            skipped_count += 1
            continue

        # 转换为16kHz单声道
        if not convert_audio_to_16k(str(temp_audio), str(final_audio)):
            logging.warning(f"Skipping {utt_id} and its corresponding text (conversion failed)")
            temp_audio.unlink(missing_ok=True)
            skipped_count += 1
            continue

        # 删除临时文件
        temp_audio.unlink(missing_ok=True)

        # 获取音频时长
        duration = get_audio_duration(str(final_audio))

        # 过滤过长或过短的音频
        if duration > max_duration:
            logging.warning(f"Skipping {utt_id} and its corresponding text (duration {duration:.2f}s > {max_duration}s)")
            final_audio.unlink(missing_ok=True)
            skipped_count += 1
            continue

        if duration < min_duration:
            logging.warning(f"Skipping {utt_id} and its corresponding text (duration {duration:.2f}s < {min_duration}s)")
            final_audio.unlink(missing_ok=True)
            skipped_count += 1
            continue

        # 创建Recording
        recording = Recording(
            id=utt_id,
            sources=[
                AudioSource(
                    type='file',
                    channels=[0],
                    source=str(final_audio.absolute())
                )
            ],
            sampling_rate=16000,
            num_samples=int(duration * 16000),
            duration=duration,
        )

        # 创建Supervision
        supervision = SupervisionSegment(
            id=utt_id,
            recording_id=utt_id,
            start=0.0,
            duration=duration,
            text=text.strip(),
            language='Chinese',
        )

        # 创建MonoCut
        cut = MonoCut(
            id=utt_id,
            start=0.0,
            duration=duration,
            channel=0,
            supervisions=[supervision],
            recording=recording,
        )

        cuts.append(cut)

        # 每100个保存一次（防止内存不足）
        if len(cuts) % 100 == 0:
            logging.info(f"Processed {len(cuts)} valid utterances so far (skipped {skipped_count})...")

    # 输出统计信息
    logging.info("=" * 60)
    logging.info(f"Processing completed!")
    logging.info(f"  Total URLs/texts: {len(urls)}")
    logging.info(f"  Successfully created: {len(cuts)} cuts")
    logging.info(f"  Skipped (timeout/failed/invalid): {skipped_count}")
    logging.info(f"  Success rate: {len(cuts)/len(urls)*100:.1f}%")
    logging.info("=" * 60)
    return cuts


def main():
    parser = argparse.ArgumentParser(description="从urls.txt和ref.txt生成Lhotse训练数据")
    parser.add_argument(
        "--urls-file",
        type=str,
        default="/workspace/bella-infra/user/zhangshuge002/master/realtime/urls.txt",
        help="音频URL文件路径"
    )
    parser.add_argument(
        "--ref-file",
        type=str,
        default="/workspace/bella-infra/user/zhangshuge002/master/realtime/ref.txt",
        help="参考文本文件路径"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/workspace/bella-infra/user/zhangshuge002/master/realtime/FireRedASR/data/fbank/custom_dataset",
        help="输出目录"
    )
    parser.add_argument(
        "--audio-dir",
        type=str,
        default=None,
        help="音频保存目录（默认为output_dir/audio）"
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=30.0,
        help="最大音频时长（秒）"
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=0.5,
        help="最小音频时长（秒）"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.9,
        help="训练集比例"
    )
    parser.add_argument(
        "--compute-fbank",
        type=int,
        default=1,
        help="是否计算Fbank特征（1=是，0=否）"
    )
    parser.add_argument(
        "--num-jobs",
        type=int,
        default=4,
        help="并行提取特征的进程数"
    )
    parser.add_argument(
        "--download-timeout",
        type=int,
        default=30,
        help="下载超时时间（秒），超过此时间会跳过该数据及对应文本"
    )

    args = parser.parse_args()
    setup_logger()

    # 读取URLs和文本
    logging.info(f"Reading URLs from {args.urls_file}")
    with open(args.urls_file, 'r', encoding='utf-8') as f:
        urls = [line.strip() for line in f if line.strip()]

    logging.info(f"Reading texts from {args.ref_file}")
    with open(args.ref_file, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]

    if len(urls) != len(texts):
        raise ValueError(f"URLs ({len(urls)}) and texts ({len(texts)}) count mismatch!")

    logging.info(f"Found {len(urls)} utterances")

    # 设置目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_dir = Path(args.audio_dir) if args.audio_dir else output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    # 创建cuts
    logging.info("Creating cuts from URLs...")
    logging.info(f"Download timeout: {args.download_timeout}s (超过此时间将跳过该数据)")
    cuts = create_cuts_from_urls(
        urls=urls,
        texts=texts,
        audio_dir=audio_dir,
        max_duration=args.max_duration,
        min_duration=args.min_duration,
        download_timeout=args.download_timeout,
    )

    if len(cuts) == 0:
        raise ValueError("No valid cuts created! Check your URLs and audio files.")

    # 创建CutSet
    cutset = CutSet.from_cuts(cuts)
    logging.info(f"Created CutSet with {len(cutset)} cuts")

    # 划分训练集和验证集
    import random
    train_size = int(len(cutset) * args.train_ratio)
    rng = random.Random(42)
    cutset_shuffled = cutset.shuffle(rng=rng)
    cuts_train = cutset_shuffled.subset(first=train_size)
    cuts_dev = cutset_shuffled.subset(last=len(cutset) - train_size)

    logging.info(f"Split: {len(cuts_train)} train, {len(cuts_dev)} dev")

    # 计算并存储Fbank特征
    if args.compute_fbank:
        logging.info("Computing Fbank features...")

        fbank_extractor = Fbank(FbankConfig(
            num_mel_bins=80,
            frame_length=0.025,
            frame_shift=0.01,
            dither=0.0,
            energy_floor=1e-10,
            raw_energy=True,
        ))

        # 训练集
        logging.info("Extracting features for training set...")
        cuts_train = cuts_train.compute_and_store_features(
            extractor=fbank_extractor,
            storage_path=str(output_dir / "feats_train"),
            num_jobs=args.num_jobs,
            storage_type=LilcomFilesWriter,
        )

        # 验证集
        logging.info("Extracting features for dev set...")
        cuts_dev = cuts_dev.compute_and_store_features(
            extractor=fbank_extractor,
            storage_path=str(output_dir / "feats_dev"),
            num_jobs=args.num_jobs,
            storage_type=LilcomFilesWriter,
        )

    # 保存CutSet
    train_manifest = output_dir / "custom_cuts_train.jsonl.gz"
    dev_manifest = output_dir / "custom_cuts_dev.jsonl.gz"

    logging.info(f"Saving training cuts to {train_manifest}")
    cuts_train.to_file(train_manifest)

    logging.info(f"Saving dev cuts to {dev_manifest}")
    cuts_dev.to_file(dev_manifest)

    # 打印统计信息
    logging.info("=" * 60)
    logging.info("Dataset Statistics:")
    logging.info(f"  Total utterances: {len(cuts)}")
    logging.info(f"  Training: {len(cuts_train)} utterances")
    logging.info(f"  Dev: {len(cuts_dev)} utterances")
    logging.info(f"  Total duration: {cutset.duration / 3600:.2f} hours")
    logging.info(f"  Training duration: {cuts_train.duration / 3600:.2f} hours")
    logging.info(f"  Dev duration: {cuts_dev.duration / 3600:.2f} hours")
    logging.info(f"  Audio directory: {audio_dir}")
    logging.info(f"  Output directory: {output_dir}")
    logging.info("=" * 60)

    logging.info("Done! You can now use these manifests for training:")
    logging.info(f"  Training manifest: {train_manifest}")
    logging.info(f"  Dev manifest: {dev_manifest}")


if __name__ == "__main__":
    main()
