# Copyright      2021  Xiaomi Corp.        (authors: Fangjun Kuang,
#                                                    Mingshuang Luo,
#                                                    Zengwei Yao)
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

"""Core utilities for training"""

import argparse
import collections
import json
import logging
import os
import pathlib
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Union

import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

Pathlike = Union[str, Path]


def str2bool(v):
    """
    Used in argparse.ArgumentParser.add_argument to indicate
    that a type is a bool type and user can enter:
        - yes, true, t, y, 1, to represent True
        - no, false, f, n, 0, to represent False
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def setup_logger(
    log_filename: Pathlike,
    log_level: str = "info",
    use_console: bool = True,
) -> None:
    """
    Setup log level.

    Args:
      log_filename:
        The filename to save the log.
      log_level:
        The log level to use, e.g., "debug", "info", "warning", "error",
        "critical"
      use_console:
        True to also print logs to console.
    """
    now = datetime.now()
    date_time = now.strftime("%Y-%m-%d-%H-%M-%S")

    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        formatter = f"%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] ({rank}/{world_size}) %(message)s"
        log_filename = f"{log_filename}-{date_time}-{rank}"
    else:
        formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
        log_filename = f"{log_filename}-{date_time}"

    os.makedirs(os.path.dirname(log_filename), exist_ok=True)

    level = logging.ERROR
    if log_level == "debug":
        level = logging.DEBUG
    elif log_level == "info":
        level = logging.INFO
    elif log_level == "warning":
        level = logging.WARNING
    elif log_level == "critical":
        level = logging.CRITICAL

    logging.basicConfig(
        filename=log_filename,
        format=formatter,
        level=level,
        filemode="w",
        force=True,
    )

    if use_console:
        console = logging.StreamHandler()
        console.setLevel(level)
        console.setFormatter(logging.Formatter(formatter))
        logging.getLogger("").addHandler(console)


class AttributeDict(dict):
    """
    A dict-like object that allows attribute access to its items.

    Example:
        >>> params = AttributeDict({"a": 1, "b": 2})
        >>> params.a
        1
        >>> params["b"]
        2
        >>> params.c = 3
        >>> params["c"]
        3
    """

    def __getattr__(self, key):
        if key in self:
            return self[key]
        raise AttributeError(f"No such attribute '{key}'")

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        if key in self:
            del self[key]
            return
        raise AttributeError(f"No such attribute '{key}'")

    def __str__(self, indent: int = 2):
        tmp = {}
        for k, v in self.items():
            # PosixPath is not JSON serializable
            if isinstance(v, (pathlib.Path, torch.device, torch.dtype)):
                v = str(v)
            tmp[k] = v
        return json.dumps(tmp, indent=indent, sort_keys=True)


class MetricsTracker(collections.defaultdict):
    """
    Track training metrics over batches.

    Example:
        >>> tracker = MetricsTracker()
        >>> tracker["loss"] = 0.5
        >>> tracker["loss"] = 0.4
        >>> tracker.norm("loss")  # Returns average
        0.45
    """

    def __init__(self):
        # Default to list for accumulating values
        super(MetricsTracker, self).__init__(list)

    def __setitem__(self, key, value):
        """Override to append values"""
        if key not in self:
            super(MetricsTracker, self).__setitem__(key, [])

        if isinstance(value, list):
            # If value is already a list, extend instead of append
            self[key].extend(value)
        elif isinstance(value, (int, float)):
            self[key].append(value)
        elif isinstance(value, torch.Tensor):
            self[key].append(value.item())
        else:
            self[key].append(float(value))

    @property
    def data(self):
        """Access underlying dict"""
        return dict(self)

    def norm(self, key):
        """Get average value for a key"""
        if key in self and len(self[key]) > 0:
            return sum(self[key]) / len(self[key])
        return 0.0

    def __add__(self, other: "MetricsTracker") -> "MetricsTracker":
        """Combine two trackers"""
        ans = MetricsTracker()
        for k, v in self.items():
            ans.data[k] = v.copy() if isinstance(v, list) else [v]
        for k, v in other.items():
            if k not in ans.data:
                ans.data[k] = []
            ans.data[k].extend(v if isinstance(v, list) else [v])
        return ans

    def __mul__(self, alpha: float) -> "MetricsTracker":
        """Scale all values"""
        ans = MetricsTracker()
        for k, v in self.items():
            ans.data[k] = [x * alpha for x in v]
        return ans

    def __str__(self) -> str:
        """String representation with averages"""
        parts = []
        for k in sorted(self.keys()):
            avg = self.norm(k)
            parts.append(f"{k}={avg:.4f}")
        return ", ".join(parts)

    def norm_items(self) -> List[Tuple[str, float]]:
        """
        Returns a list of pairs with normalized (averaged) values.

        Returns:
            List of (key, average_value) tuples
        """
        ans = []
        for k in self.keys():
            ans.append((k, self.norm(k)))
        return ans

    def reduce(self, device):
        """
        Reduce metrics across distributed processes using torch.distributed.

        Args:
            device: Device to place tensors on
        """
        if not (dist.is_available() and dist.is_initialized()):
            return

        keys = sorted(self.keys())
        # Convert to tensor for reduction
        values = []
        counts = []
        for k in keys:
            if len(self[k]) > 0:
                values.append(sum(self[k]))
                counts.append(len(self[k]))
            else:
                values.append(0.0)
                counts.append(0)

        values_tensor = torch.tensor(values, device=device)
        counts_tensor = torch.tensor(counts, device=device)

        # All-reduce sum
        dist.all_reduce(values_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(counts_tensor, op=dist.ReduceOp.SUM)

        # Update with reduced values
        for k, v, c in zip(keys, values_tensor.cpu().tolist(), counts_tensor.cpu().tolist()):
            if c > 0:
                # Store as single averaged value
                self.data[k] = [v / c]
            else:
                self.data[k] = []

    def write_summary(
        self,
        tb_writer: SummaryWriter,
        prefix: str,
        batch_idx: int,
    ) -> None:
        """
        Add logging information to a TensorBoard writer.

        Args:
            tb_writer: A TensorBoard SummaryWriter
            prefix: A prefix for the metric name, e.g. "train/valid_" or "train/"
            batch_idx: The current batch index, used as the x-axis of the plot
        """
        for k, v in self.norm_items():
            tb_writer.add_scalar(prefix + k, v, batch_idx)
