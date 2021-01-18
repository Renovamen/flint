import .functional as F
from .tensor import Tensor

class Loss:
    def __init__(self, reduction: str = 'mean') -> None:
        super(Loss, self).__init__()
        self.reduction = reduction


class NllLoss(Loss):
    '''
    Negative Log Likelihood Loss
    See tinyark.functional.nll_loss() for more details.

    args:
        reduction (str, optional): 'none' / 'mean' / 'sum'
    '''
    def __init__(self, reduction: str = 'mean') -> None:
        super(NllLoss, self).__init__(reduction)

    def __call__(self, input: Tensor, target: Tensor) -> Tensor:
        self.data = F.nll_loss(input, target, reduction = self.reduction, out = self.data)
        return self.data


class CrossEntropyLoss(Loss):
    '''
    Cross Entropy Loss, combines log_softmax() and nll_loss().
    See tinyark.functional.cross_entropy() for more details.

    args:
        reduction (str, optional): 'none' / 'mean' / 'sum'
    '''
    def __init__(self, reduction: str = 'mean') -> None:
        super(CrossEntropyLoss, self).__init__(reduction)

    def __call__(self, input: Tensor, target: Tensor) -> Tensor:
        self.data = F.cross_entropy(input, target, reduction = self.reduction, out = self.data)
        return self.data
