from tinyark import Tensor
from .. import functional as F

class Loss:
    def __init__(self, reduction: str = 'mean') -> None:
        super(Loss, self).__init__()
        self.reduction = reduction

class NllLoss(Loss):
    '''
    Negative Log Likelihood Loss
    See tinyark.nn.functional.nll_loss() for more details.

    args:
        reduction (str, optional): 'none' / 'mean' / 'sum'
    '''

    def __init__(self, reduction: str = 'mean') -> None:
        super(NllLoss, self).__init__(reduction)

    def __call__(self, input: Tensor, target: Tensor) -> Tensor:
        self.data = F.nll_loss(input, target, reduction=self.reduction)
        return self.data

class CrossEntropyLoss(Loss):
    '''
    Cross Entropy Loss, combines softmax() and nll_loss().
    See tinyark.nn.functional.cross_entropy() for more details.

    args:
        reduction (str, optional): 'none' / 'mean' / 'sum'
    '''

    def __init__(self, reduction: str = 'mean') -> None:
        super(CrossEntropyLoss, self).__init__(reduction)

    def __call__(self, input: Tensor, target: Tensor) -> Tensor:
        self.data = F.cross_entropy(input, target, reduction=self.reduction)
        return self.data
