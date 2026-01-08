# Copyright      2021  Xiaomi Corp.        (authors: Fangjun Kuang)
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

"""Distributed training utilities"""

import os

import torch
from torch import distributed as dist


def setup_dist(
    rank=None, world_size=None, master_port=None, use_ddp_launch=False, master_addr=None
):
    """
    Setup distributed training.

    Args:
        rank: Process rank (only used if use_ddp_launch=False)
        world_size: Total number of processes (only used if use_ddp_launch=False)
        master_port: Master port for communication
        use_ddp_launch: Whether using torchrun/torch.distributed.launch
        master_addr: Master address for communication
    """
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = (
            "localhost" if master_addr is None else str(master_addr)
        )

    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "12354" if master_port is None else str(master_port)

    if use_ddp_launch is False:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
    else:
        dist.init_process_group("nccl")


def cleanup_dist():
    """Cleanup distributed training"""
    dist.destroy_process_group()


def get_world_size():
    """Get world size (total number of processes)"""
    if "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"])
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    else:
        return 1


def get_rank():
    """Get current process rank"""
    if "RANK" in os.environ:
        return int(os.environ["RANK"])
    elif dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    else:
        return 0


def get_local_rank():
    """Get local rank within a node"""
    return int(os.environ.get("LOCAL_RANK", 0))
