# adopted from: https://github.com/pytorch/pytorch/blob/master/torch/utils/data/dataset.py

import bisect
from typing import List, Iterable

class Dataset:
    def __getitem__(self, index):
        raise NotImplementedError

    def __add__(self, other: 'Dataset') -> 'ConcatDataset':
        return ConcatDataset([self, other])

class ConcatDataset(Dataset):
    '''
    Dataset as a concatenation of multiple datasets.
    This class is useful to assemble different existing datasets.

    args:
        datasets (sequence): List of datasets to be concatenated
    '''

    datasets: List[Dataset]
    cumulative_sizes: List[int]

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets: Iterable[Dataset]) -> None:
        super(ConcatDataset, self).__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]
