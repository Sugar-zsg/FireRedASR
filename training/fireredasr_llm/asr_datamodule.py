# Copyright      2021  Piotr Żelasko
# Copyright      2022  Xiaomi Corporation     (Author: Mingshuang Luo)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import inspect
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from lhotse import CutSet, Fbank, FbankConfig, load_manifest, load_manifest_lazy
from lhotse.dataset import (
    CutMix,
    DynamicBucketingSampler,
    K2SpeechRecognitionDataset,
    PrecomputedFeatures,
    SimpleCutSampler,
    SpecAugment,
)
from lhotse.utils import fix_random_seed
from torch.utils.data import DataLoader


def str2bool(v):
    """Used in argparse to parse boolean arguments"""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


class _SeedWorkers:
    def __init__(self, seed: int):
        self.seed = seed

    def __call__(self, worker_id: int):
        fix_random_seed(self.seed + worker_id)


class AsrDataModule:
    """
    DataModule for FireRedASR-LLM training with Lhotse.

    It handles:
    - Dynamic batch sizing with bucketing
    - Data augmentation (SpecAugment, MUSAN noise mixing)
    - Precomputed fbank features loading
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(
            title="ASR data related options",
            description="Options for PyTorch DataLoaders from Lhotse CutSets",
        )
        group.add_argument(
            "--manifest-dir",
            type=Path,
            default=Path("data/fbank"),
            help="Path to directory with train/valid/test cuts.",
        )
        group.add_argument(
            "--max-duration",
            type=float,
            default=200.0,
            help="Maximum pooled recordings duration (seconds) in a "
            "single batch. Reduce if CUDA OOM occurs.",
        )
        group.add_argument(
            "--bucketing-sampler",
            type=str2bool,
            default=True,
            help="Use bucketing sampler (saves padding frames).",
        )
        group.add_argument(
            "--num-buckets",
            type=int,
            default=15,
            help="Number of buckets for DynamicBucketingSampler.",
        )
        group.add_argument(
            "--shuffle",
            type=str2bool,
            default=True,
            help="Shuffle examples for each epoch.",
        )
        group.add_argument(
            "--drop-last",
            type=str2bool,
            default=True,
            help="Whether to drop last batch in sampler.",
        )
        group.add_argument(
            "--return-cuts",
            type=str2bool,
            default=True,
            help="Return cuts in batch['supervisions']['cut'].",
        )
        group.add_argument(
            "--num-workers",
            type=int,
            default=4,
            help="Number of dataloader workers.",
        )
        group.add_argument(
            "--enable-spec-aug",
            type=str2bool,
            default=True,
            help="Enable SpecAugment for training.",
        )
        group.add_argument(
            "--spec-aug-time-warp-factor",
            type=int,
            default=80,
            help="Time warp factor for SpecAugment. <1 disables time warp.",
        )
        group.add_argument(
            "--enable-musan",
            type=str2bool,
            default=True,
            help="Enable MUSAN noise mixing for training.",
        )
        group.add_argument(
            "--cmvn-path",
            type=Path,
            default=None,
            help="Path to CMVN statistics file (cmvn.ark). If provided, CMVN will be applied during training.",
        )

    def train_dataloaders(
        self,
        cuts_train: CutSet,
        sampler_state_dict: Optional[Dict[str, Any]] = None,
    ) -> DataLoader:
        """
        Create training DataLoader with augmentation.

        Args:
          cuts_train: CutSet for training
          sampler_state_dict: State dict for resuming sampler
        """
        transforms = []

        # MUSAN noise mixing
        if self.args.enable_musan:
            logging.info("Enable MUSAN noise mixing")
            cuts_musan = load_manifest(self.args.manifest_dir / "musan_cuts.jsonl.gz")
            transforms.append(
                CutMix(cuts=cuts_musan, p=0.5, snr=(10, 20), preserve_id=True)
            )
        else:
            logging.info("Disable MUSAN")

        # CMVN normalization
        input_transforms = []
        if hasattr(self.args, 'cmvn_path') and self.args.cmvn_path is not None and self.args.cmvn_path.exists():
            logging.info(f"✓ [TRAIN] Enable CMVN normalization from {self.args.cmvn_path}")
            print(f"[TRAIN] Loading CMVN statistics from: {self.args.cmvn_path}", flush=True)
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent.parent.parent))
            from fireredasr.data.asr_feat import CMVN

            cmvn = CMVN(str(self.args.cmvn_path))
            print(f"[TRAIN] CMVN loaded successfully: dim={cmvn.dim}, will normalize features to mean≈0, std≈1", flush=True)

            # Create a wrapper to make CMVN compatible with Lhotse
            class CMVNTransform:
                def __init__(self, cmvn_obj):
                    self.cmvn = cmvn_obj

                def __call__(self, features, **kwargs):
                    # features: (T, F) numpy array or torch tensor
                    # kwargs: Lhotse may pass supervision_segments and other args, we ignore them
                    import torch
                    import numpy as np

                    is_tensor = isinstance(features, torch.Tensor)
                    if is_tensor:
                        device = features.device
                        dtype = features.dtype
                        features = features.cpu().numpy()

                    # Apply CMVN
                    normalized = self.cmvn(features)

                    if is_tensor:
                        normalized = torch.from_numpy(normalized).to(device=device, dtype=dtype)

                    return normalized

            input_transforms.append(CMVNTransform(cmvn))
        else:
            if not hasattr(self.args, 'cmvn_path') or self.args.cmvn_path is None:
                logging.info("✗ [TRAIN] CMVN not provided, skipping CMVN normalization")
                print(f"[TRAIN] CMVN normalization: DISABLED (no --cmvn-path provided)", flush=True)
            else:
                logging.warning(f"✗ [TRAIN] CMVN file not found at {self.args.cmvn_path}, skipping CMVN normalization")
                print(f"[TRAIN] WARNING: CMVN file not found at {self.args.cmvn_path}", flush=True)

        # SpecAugment
        if self.args.enable_spec_aug:
            logging.info("Enable SpecAugment")
            logging.info(f"Time warp factor: {self.args.spec_aug_time_warp_factor}")

            # Check Lhotse version for num_frame_masks default
            num_frame_masks = 10
            num_frame_masks_parameter = inspect.signature(
                SpecAugment.__init__
            ).parameters["num_frame_masks"]
            if num_frame_masks_parameter.default == 1:
                num_frame_masks = 2
            logging.info(f"Num frame masks: {num_frame_masks}")

            input_transforms.append(
                SpecAugment(
                    time_warp_factor=self.args.spec_aug_time_warp_factor,
                    num_frame_masks=num_frame_masks,
                    features_mask_size=27,
                    num_feature_masks=2,
                    frames_mask_size=100,
                )
            )
        else:
            logging.info("Disable SpecAugment")

        # Create dataset with precomputed features
        logging.info("Creating train dataset with PrecomputedFeatures")
        train = K2SpeechRecognitionDataset(
            input_strategy=PrecomputedFeatures(),
            cut_transforms=transforms,
            input_transforms=input_transforms,
            return_cuts=self.args.return_cuts,
        )

        # Create sampler
        if self.args.bucketing_sampler:
            logging.info("Using DynamicBucketingSampler")
            train_sampler = DynamicBucketingSampler(
                cuts_train,
                max_duration=self.args.max_duration,
                shuffle=self.args.shuffle,
                num_buckets=self.args.num_buckets,
                buffer_size=self.args.num_buckets * 2000,
                shuffle_buffer_size=self.args.num_buckets * 5000,
                drop_last=self.args.drop_last,
            )
            shuffle_status = "✓ ENABLED" if self.args.shuffle else "✗ DISABLED"
            print(f"[TRAIN] Data Shuffling: {shuffle_status}", flush=True)
            print(f"[TRAIN]   - Sampler: DynamicBucketingSampler", flush=True)
            print(f"[TRAIN]   - Num buckets: {self.args.num_buckets}", flush=True)
            print(f"[TRAIN]   - Shuffle buffer size: {self.args.num_buckets * 5000}", flush=True)
            logging.info(f"Training data shuffle: {self.args.shuffle}")
        else:
            logging.info("Using SimpleCutSampler")
            train_sampler = SimpleCutSampler(
                cuts_train,
                max_duration=self.args.max_duration,
                shuffle=self.args.shuffle,
            )
            shuffle_status = "✓ ENABLED" if self.args.shuffle else "✗ DISABLED"
            print(f"[TRAIN] Data Shuffling: {shuffle_status}", flush=True)
            print(f"[TRAIN]   - Sampler: SimpleCutSampler", flush=True)
            logging.info(f"Training data shuffle: {self.args.shuffle}")

        # Load sampler state if resuming
        if sampler_state_dict is not None:
            logging.info("Loading sampler state dict")
            train_sampler.load_state_dict(sampler_state_dict)

        # Set random seed for workers
        seed = torch.randint(0, 100000, ()).item()
        worker_init_fn = _SeedWorkers(seed)

        # Create DataLoader
        logging.info("Creating train dataloader")
        train_dl = DataLoader(
            train,
            sampler=train_sampler,
            batch_size=None,
            num_workers=self.args.num_workers,
            persistent_workers=True if self.args.num_workers > 0 else False,
            worker_init_fn=worker_init_fn,
        )

        return train_dl

    def valid_dataloaders(self, cuts_valid: CutSet) -> DataLoader:
        """
        Create validation DataLoader without augmentation.

        Args:
          cuts_valid: CutSet for validation
        """
        logging.info("Creating validation dataset")

        # CMVN normalization for validation (same as training)
        input_transforms = []
        if hasattr(self.args, 'cmvn_path') and self.args.cmvn_path is not None and self.args.cmvn_path.exists():
            logging.info(f"✓ [VALID] Enable CMVN normalization for validation from {self.args.cmvn_path}")
            print(f"[VALID] Loading CMVN statistics from: {self.args.cmvn_path}", flush=True)
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent.parent.parent))
            from fireredasr.data.asr_feat import CMVN

            cmvn = CMVN(str(self.args.cmvn_path))
            print(f"[VALID] CMVN loaded successfully for validation", flush=True)

            class CMVNTransform:
                def __init__(self, cmvn_obj):
                    self.cmvn = cmvn_obj

                def __call__(self, features, **kwargs):
                    # kwargs: Lhotse may pass supervision_segments and other args, we ignore them
                    import torch
                    import numpy as np

                    is_tensor = isinstance(features, torch.Tensor)
                    if is_tensor:
                        device = features.device
                        dtype = features.dtype
                        features = features.cpu().numpy()

                    normalized = self.cmvn(features)

                    if is_tensor:
                        normalized = torch.from_numpy(normalized).to(device=device, dtype=dtype)

                    return normalized

            input_transforms.append(CMVNTransform(cmvn))

        validate = K2SpeechRecognitionDataset(
            input_strategy=PrecomputedFeatures(),
            input_transforms=input_transforms if input_transforms else None,
            return_cuts=self.args.return_cuts,
        )

        # Use fewer buckets for small validation sets
        num_valid_cuts = len(cuts_valid)
        num_buckets = min(self.args.num_buckets, max(1, num_valid_cuts // 2))

        valid_sampler = DynamicBucketingSampler(
            cuts_valid,
            max_duration=self.args.max_duration,
            shuffle=False,
            num_buckets=num_buckets,
        )

        logging.info("Creating validation dataloader")
        valid_dl = DataLoader(
            validate,
            sampler=valid_sampler,
            batch_size=None,
            num_workers=2,
            persistent_workers=False,
        )

        return valid_dl

    def test_dataloaders(self, cuts: CutSet) -> DataLoader:
        """
        Create test DataLoader.

        Args:
          cuts: CutSet for testing
        """
        logging.info("Creating test dataset")
        test = K2SpeechRecognitionDataset(
            input_strategy=PrecomputedFeatures(),
            return_cuts=self.args.return_cuts,
        )

        sampler = DynamicBucketingSampler(
            cuts,
            max_duration=self.args.max_duration,
            shuffle=False,
        )

        logging.info("Creating test dataloader")
        test_dl = DataLoader(
            test,
            batch_size=None,
            sampler=sampler,
            num_workers=self.args.num_workers,
        )

        return test_dl
