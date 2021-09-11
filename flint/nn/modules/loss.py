from flint import Tensor
from .module import Module
from .. import functional as F

__all__ = [
    'Loss',
    'NllLoss',
    'CrossEntropyLoss',
    'MSELoss',
    'BCELoss'
]


class Loss(Module):
    def __init__(self, reduction: str = 'mean') -> None:
        super(Loss, self).__init__()
        self.reduction = reduction


class NllLoss(Loss):
    """
    Negative Log Likelihood Loss. See :func:`flint.nn.functional.nll_loss`
    for more details.

    Args:
        reduction (str, optional): 'none' / 'mean' / 'sum'
    """
    def __init__(self, reduction: str = 'mean') -> None:
        super(NllLoss, self).__init__(reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        self.data = F.nll_loss(input, target, reduction=self.reduction)
        return self.data


class CrossEntropyLoss(Loss):
    """
    Cross Entropy Loss, combines :func:`~flint.Tensor.softmax`
    and :func:`~flint.nn.functional.nll_loss`. See
    :func:`flint.nn.functional.cross_entropy` for more details.

    Args:
        reduction (str, optional): 'none' / 'mean' / 'sum'
    """
    def __init__(self, reduction: str = 'mean') -> None:
        super(CrossEntropyLoss, self).__init__(reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        self.data = F.cross_entropy(input, target, reduction=self.reduction)
        return self.data


class MSELoss(Loss):
    """
    Mean Squared Error Loss: :math:`(x - y)^2`. See :func:`flint.nn.functional.mse_loss`
    for more details.

    Args:
        reduction (str, optional): 'none' / 'mean' / 'sum'
    """
    def __init__(self, reduction: str = 'mean') -> None:
        super(MSELoss, self).__init__(reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        self.data = F.mse_loss(input, target, reduction=self.reduction)
        return self.data


class BCELoss(Loss):
    """
    Binary Cross Entropy Loss:

    .. math::
        \\text{loss} = y \log(x) + (1 - y) \log(1 - x)

    See :func:`flint.nn.functional.binary_cross_entropy` for more details.

    Args:
        reduction (str, optional): 'none' / 'mean' / 'sum'
    """
    def __init__(self, reduction: str = 'mean') -> None:
        super(BCELoss, self).__init__(reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        self.data = F.binary_cross_entropy(input, target, reduction=self.reduction)
        return self.data
