import numpy as np

from .tensor import Tensor

__all__ = ['eq']

def eq(input: Tensor, target: Tensor) -> Tensor:
    if input.shape != target.shape:
        raise ValueError(
            "Expected input shape ({}) to match target shape ({}).".format(input.shape, target.shape)
        )
    return Tensor(np.equal(input.data, target.data))
