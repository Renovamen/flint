import numpy as np
import math
from typing import Any

from flint import Tensor
from .module import Module
from .. import Parameter, init
from .. import functional as F


class Identity(Module):
    """
    A placeholder identity operator that is argument-insensitive.

    - Input shape: :math:`(*)`, where :math:`*` means any number of dimensions.
    - Output shape: :math:`(*)`, same shape as the input.

    Parameters
    ----------
    args : Any
        Any argument (unused).

    kwargs: Any
        Any keyword argument (unused).
    """
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super(Identity, self).__init__()

    def forward(self, input: Tensor) -> Tensor:
        return input


class Linear(Module):
    """
    Full connected layer

    .. math::
        y = x A^T + b

    - Input shape: ``(batch_size, in_features)``
    - Output shape: ``(batch_size, out_features)``

    Parameters
    ----------
    in_features : int
        Size of each input sample.

    out_features : int
        Size of each output sample.

    bias : bool, optional, default=True
        Enable bias or not.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(Linear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(Tensor.zeros(in_features, out_features))
        if bias:
            self.bias = Parameter(Tensor.zeros(1, out_features))
        else:
            self.register_parameter('bias', None)

        self.init_parameters()

    def init_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        self.output = F.linear(input, self.weight, self.bias)
        return self.output
