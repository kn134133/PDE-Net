# Copyright (c) OpenMMLab. All rights reserved.
import itertools
import json
import math
import os
import random
from typing import Iterator, Optional, Sized

import torch
import torch.distributed as dist
from torch.utils.data import Sampler

from mmengine.dist import get_dist_info, sync_random_seed
from mmengine.registry import DATA_SAMPLERS

@DATA_SAMPLERS.register_module()
class SPDefaultSampler(Sampler):
    """The default data sampler for both distributed and non-distributed
    environment.

    It has several differences from the PyTorch ``DistributedSampler`` as
    below:

    1. This sampler supports non-distributed environment.

    2. The round up behaviors are a little different.

       - If ``round_up=True``, this sampler will add extra samples to make the
         number of samples is evenly divisible by the world size. And
         this behavior is the same as the ``DistributedSampler`` with
         ``drop_last=False``.
       - If ``round_up=False``, this sampler won't remove or add any samples
         while the ``DistributedSampler`` with ``drop_last=True`` will remove
         tail samples.

    Args:
        dataset (Sized): The dataset.
        shuffle (bool): Whether shuffle the dataset or not. Defaults to True.
        seed (int, optional): Random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Defaults to None.
        round_up (bool): Whether to add extra samples to make the number of
            samples evenly divisible by the world size. Defaults to True.
    """

    def __init__(self,
                 dataset: Sized,
                 shuffle: bool = True,
                 seed: Optional[int] = None
                 ) -> None:

        self.dataset = dataset
        self.shuffle = shuffle
        if seed is None:
            seed = sync_random_seed()
        self.seed = seed
        self.epoch = 0
        self.size = len(dataset)
        
    def __iter__(self) -> Iterator[int]:
        """Iterate the indices."""
        # deterministically shuffle based on epoch and seed
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(self.size, generator=g).tolist()
        else:
            indices = torch.arange(self.size).tolist()

        return iter(indices)

    def __len__(self) -> int:
        """The number of samples in this rank."""
        return self.size

    def set_epoch(self, epoch: int) -> None:
        """Sets the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas use a different
        random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch

@DATA_SAMPLERS.register_module()
class SPInfiniteSampler(Sampler):
    """It's designed for iteration-based runner and yields a mini-batch indices
    each time.

    The implementation logic is referred to
    https://github.com/facebookresearch/detectron2/blob/main/detectron2/data/samplers/distributed_sampler.py

    Args:
        dataset (Sized): The dataset.
        shuffle (bool): Whether shuffle the dataset or not. Defaults to True.
        seed (int, optional): Random seed. If None, set a random seed.
            Defaults to None.
    """  # noqa: W605

    def __init__(self,
                 dataset: Sized,
                 shuffle: bool = True,
                 seed: Optional[int] = None) -> None:

        self.dataset = dataset
        self.shuffle = shuffle
        if seed is None:
            seed = sync_random_seed()
        self.seed = seed
        self.size = len(dataset)
        self.indices = self._indices()

    def _infinite_indices(self) -> Iterator[int]:
        """Infinitely yield a sequence of indices."""
        g = torch.Generator()
        g.manual_seed(self.seed)
        while True:
            if self.shuffle:
                yield from torch.randperm(self.size, generator=g).tolist()

            else:
                yield from torch.arange(self.size).tolist()

    def _indices(self) -> Iterator[int]:
        """Slice the infinite indices by rank."""
        yield from itertools.islice(self._infinite_indices(), 0, None, 1)

    def __iter__(self) -> Iterator[int]:
        """Iterate the indices."""
        yield from self.indices

    def __len__(self) -> int:
        """Length of base dataset."""
        return self.size

    def set_epoch(self, epoch: int) -> None:
        """Not supported in iteration-based runner."""
        pass


@DATA_SAMPLERS.register_module()
class SPPairSampler(Sampler):
    """Spatial-pair sampler for REST spatial-parallel training.

    Each iteration, GPU-rank-0 gets index of patch A and GPU-rank-1 gets
    index of the spatially adjacent patch B (from the same source image).
    This ensures the all2all communication in ASADA exchanges meaningful
    spatial context rather than features from unrelated images.

    Args:
        dataset (Sized): The dataset. Must have ``data_list`` attribute
            where each item has ``img_path`` containing the patch filename.
        pair_index_file (str): Path to JSON file containing list of
            [patchA_filename, patchB_filename] pairs.
        shuffle (bool): Whether to shuffle pairs each epoch. Defaults to True.
        seed (int, optional): Random seed. Defaults to None.
    """

    def __init__(self,
                 dataset: Sized,
                 pair_index_file: str,
                 shuffle: bool = True,
                 seed: Optional[int] = None) -> None:

        self.dataset = dataset
        self.shuffle = shuffle
        if seed is None:
            seed = sync_random_seed()
        self.seed = seed

        # Build filename -> dataset index mapping
        self.fname_to_idx = {}
        for idx in range(len(dataset)):
            data_info = dataset.get_data_info(idx)
            fname = os.path.basename(data_info['img_path'])
            self.fname_to_idx[fname] = idx

        # Load pairs and resolve to dataset indices
        with open(pair_index_file, 'r') as f:
            raw_pairs = json.load(f)

        self.pairs = []
        skipped = 0
        for pa, pb in raw_pairs:
            if pa in self.fname_to_idx and pb in self.fname_to_idx:
                self.pairs.append((self.fname_to_idx[pa], self.fname_to_idx[pb]))
            else:
                skipped += 1
        if skipped:
            print(f'[SPPairSampler] Skipped {skipped} pairs (not found in dataset)')
        print(f'[SPPairSampler] Loaded {len(self.pairs)} valid pairs')

        # Get rank
        if dist.is_available() and dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1

        self._pair_indices = self._make_indices()

    def _make_indices(self) -> Iterator[int]:
        """Infinitely yield patch indices according to rank."""
        g = torch.Generator()
        g.manual_seed(self.seed)
        while True:
            if self.shuffle:
                order = torch.randperm(len(self.pairs), generator=g).tolist()
            else:
                order = list(range(len(self.pairs)))
            for pair_idx in order:
                pa_idx, pb_idx = self.pairs[pair_idx]
                # rank 0 -> patch A, rank 1 -> patch B
                if self.rank % 2 == 0:
                    yield pa_idx
                else:
                    yield pb_idx

    def __iter__(self) -> Iterator[int]:
        yield from self._pair_indices

    def __len__(self) -> int:
        return len(self.pairs)

    def set_epoch(self, epoch: int) -> None:
        pass