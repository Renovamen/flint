import numpy as np
from typing import Union
from .tensor import Tensor
from .utils import *

def nll_loss(
    input: Tensor,
    target: Tensor,
    reduction: str = 'mean'
) -> Tensor:
    '''
    Negative Log Likelihood Loss

    NOTE: Here I apply log() on the prediction data, which is DIFFERENT FROM
          F.nll_loss() IN PYTORCH!

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
    delta = 1e-7  # deal with the situation that input.data = 0

    ret = - np.log(input.data[np.arange(batch_size), target.data.astype(np.int)] + delta)
    if reduction in ['sum', 'mean']:
        ret = np.sum(ret)
    if reduction == 'mean':
        ret = ret / batch_size
    
    out = Tensor(
        data = ret,
        depends_on = [input],
        requires_grad = input.requires_grad
    )

    def grad_nll():
        if input.requires_grad:
            y = to_categorical(target.data)
            if reduction == 'mean':
                input.grad += (input.data - y) / batch_size  # (batch_size, n_classes)
            elif reduction == 'sum':
                input.grad += (input.data - y)  # (batch_size, n_classes)

    if out.requires_grad and reduction != 'none':
        out.grad_fn = grad_nll

    return out

def cross_entropy(
    input: Tensor,
    target: Tensor,
    reduction: str = 'mean'
) -> Tensor:
    '''
    Cross Entropy Loss
    
    NOTE: Combines softmax() and nll_loss(), which is DIFFERENT FROM
          F.cross_entropy() IN PYTORCH!

    args:
        input (Tensor): 2-dim (N, C), where N = batch size and C = number of classes
        target (Tensor): 1-dim (N) where each value: 0 <= target[i] <= C-1
        reduction (str, optional): 'none' / 'mean' / 'sum'
    '''

    after_softmax = input.softmax(axis=-1)
    out = nll_loss(after_softmax, target, reduction)

    return out
