import os
import sys
import torchvision
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import flint
from flint import Tensor

def collate_fn(batch):
    elem = batch[0]
    if isinstance(elem, int):
        return Tensor(np.array(batch))
    elif isinstance(elem, tuple):
        return tuple(collate_fn(samples) for samples in zip(*batch))
    else:
        return Tensor(np.stack([samples.numpy() for samples in batch]))

def get_data(batch_size: int):
    trans = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (1.0,))
    ])

    train_data = torchvision.datasets.MNIST(
        root = "./data/",
        transform = trans,
        train = True,
        download = True
    )
    test_data = torchvision.datasets.MNIST(
        root = "./data/",
        transform = trans,
        train = False
    )

    train_loader = flint.utils.data.DataLoader(
        dataset = train_data,
        batch_size = batch_size,
        collate_fn = collate_fn
    )
    test_loader = flint.utils.data.DataLoader(
        dataset = test_data,
        batch_size = batch_size,
        collate_fn = collate_fn
    )

    return train_loader, test_loader
