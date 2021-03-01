# adopted from: https://github.com/teddykoker/tinyloader/blob/main/dataloader.py

import math
from typing import Optional, Any, Callable, List
from .dataset import Dataset
from . import _utils

class DataLoader:
    """
    DataLoader provides an iterable over the given dataset. It supports
    automatic mini-batching now.

    Args:
        dataset (Dataset): Dataset from which to load the data
        batch_size (int, optional, default=1): How many samples per batch to load
        collate_fn (callable, optional): Merge a list of samples to form a
            mini-batch of Tensor(s)
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        collate_fn: Optional[Callable] = None
    ):
        self.index = 0
        self.dataset = dataset
        self.batch_size = batch_size

        if collate_fn is None:
            collate_fn = _utils.default_collate

        self.collate_fn = collate_fn

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self) -> Any:
        if self.index >= len(self.dataset):
            raise StopIteration
        batch_size = min(len(self.dataset) - self.index, self.batch_size)
        return self.collate_fn([self.get() for _ in range(batch_size)])

    def __len__(self) -> int:
        return math.ceil(len(self.dataset) / self.batch_size)

    def get(self):
        item = self.dataset[self.index]
        self.index += 1
        return item
