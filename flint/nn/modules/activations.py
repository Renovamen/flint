from flint import Tensor
from .. import functional as F
from .module import Module

__all__ = [
    'ReLU',
    'Sigmoid',
    'Tanh'
]


class ReLU(Module):
    """
    ReLU (Rectified Linear Unit) activation function. See
    :func:`flint.nn.functional.relu` for more details.
    """
    def __init__(self) -> None:
        super(ReLU, self).__init__()

    def forward(self, input: Tensor) -> Tensor:
        self.data = F.relu(input)
        return self.data

class Sigmoid(Module):
    """
    Sigmoid activation function. See :func:`flint.nn.functional.sigmoid`
    for more details.
    """
    def __init__(self) -> None:
        super(Sigmoid, self).__init__()

    def forward(self, input: Tensor) -> Tensor:
        self.data = F.sigmoid(input)
        return self.data

class Tanh(Module):
    """
    Tanh (Hyperbolic Tangent) activation function. See
    :func:`flint.nn.functional.tanh` for more details.
    """
    def __init__(self) -> None:
        super(Tanh, self).__init__()

    def forward(self, input: Tensor) -> Tensor:
        self.data = F.tanh(input)
        return self.data
