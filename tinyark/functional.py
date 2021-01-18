import numpy as np
from numpy import ndarray
from .tensor import Tensor

def softmax(x: Tensor, dim: int = -1, out: Optional[Tensor] = None) -> Tensor:
    if out is None:
        out = Tensor(shape = x.shape)
    np.subtract(x.data, np.max(x.data, axis = dim, keepdims = True), out = out.data)
    np.exp(out.data, out = out.data)
    np.divide(out.data, np.sum(x.data, axis = dim, keepdims = True), out = out.data)
    return out

def log_softmax(x: Tensor, dim: int = -1, out: Optional[Tensor] = None) -> Tensor:
    if out is None:
        out = Tensor(shape = x.shape)
    softmax(x, dim, out)
    np.log(out.data, out = out.data)
    return out

def nll_loss(
    input: Tensor,
    target: Tensor,
    reduction: str = 'mean',
    out: Optional[Tensor] = None
) -> Tensor:
    '''
    Negative Log Likelihood Loss

    args:
        input (Tensor): 2-dim (N, C), where N = batch size and C = number of classes
        target (Tensor): 1-dim (N) where each value: 0 <= target[i] <= C-1
        reduction (str, optional): 'none' / 'mean' / 'sum'
    '''

    dim = input.ndim

    if dim != 2:
        raise ValueError("Expected 2 dimensions (got {})".format(dim))

    if input.shape[0] != target.shape[0]:
        raise ValueError(
            "Expected input batch_size ({}) to match target batch_size ({}).".format(input.shape[0], target.shape[0])
        )

    batch_size = input.shape[0]
    n_classes = input.shape[1]

    ret = np.multiply(-1, x.data[np.arange(batch_size), target.data.astype(np.int)])
    if reduction in ['sum', 'mean']:
        ret = np.sum(ret)
    if reduction == 'mean':
        ret = np.divide(ret, batch_size)

    if out is None:
        out = Tensor(shape = ret.shape)
    out.data = ret

    return out

def cross_entropy(
    input: Tensor,
    target: Tensor,
    reduction: str = 'mean',
    out: Tensor = None
) -> Tensor:
    '''
    Cross Entropy Loss
    combines log_softmax() and nll_loss()

    args:
        input (Tensor): 2-dim (N, C), where N = batch size and C = number of classes
        target (Tensor): 1-dim (N) where each value: 0 <= target[i] <= C-1
        reduction (str, optional): 'none' / 'mean' / 'sum'
    '''

    ret = log_softmax(input, dim = 1)
    nll_loss(ret, target, reduction, out)

    return out
