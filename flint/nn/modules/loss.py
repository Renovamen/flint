from flint import Tensor
from .. import functional as F

class Loss:
    def __init__(self, reduction: str = 'mean') -> None:
        super(Loss, self).__init__()
        self.reduction = reduction

class NllLoss(Loss):
    '''
    Negative Log Likelihood Loss
    See flint.nn.functional.nll_loss() for more details.

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
    See flint.nn.functional.cross_entropy() for more details.

    args:
        reduction (str, optional): 'none' / 'mean' / 'sum'
    '''

    def __init__(self, reduction: str = 'mean') -> None:
        super(CrossEntropyLoss, self).__init__(reduction)

    def __call__(self, input: Tensor, target: Tensor) -> Tensor:
        self.data = F.cross_entropy(input, target, reduction=self.reduction)
        return self.data

class MSELoss(Loss):
    '''
    Mean Squared Error Loss: (x - y)^2
    See flint.nn.functional.mse_loss() for more details.

    args:
        reduction (str, optional): 'none' / 'mean' / 'sum'
    '''

    def __init__(self, reduction: str = 'mean') -> None:
        super(MSELoss, self).__init__(reduction)

    def __call__(self, input: Tensor, target: Tensor) -> Tensor:
        self.data = F.mse_loss(input, target, reduction=self.reduction)
        return self.data

class BCELoss(Loss):
    '''
    Binary Cross Entropy Loss:
        loss = y * log(x) + (1 - y) * log(1 - x)
    See flint.nn.functional.bce_loss() for more details.

    args:
        reduction (str, optional): 'none' / 'mean' / 'sum'
    '''

    def __init__(self, reduction: str = 'mean') -> None:
        super(BCELoss, self).__init__(reduction)

    def __call__(self, input: Tensor, target: Tensor) -> Tensor:
        self.data = F.binary_cross_entropy(input, target, reduction=self.reduction)
        return self.data
