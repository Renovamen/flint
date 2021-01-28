import numpy as np
from .optimizer import Optimizer

class Adagrad(Optimizer):
    '''
    Adagrad:
        h_t = h_{t-1} + (g_t)^2
        p_{t+1} = p_t - lr / sqrt(h_t + eps) * g_t

    args:
        params (iterable): an iterable of Tensor
        lr (float, optional): learning rate (default: 0.01)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-10)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    '''

    def __init__(
        self,
        params = None,
        lr: float = 0.01,
        eps: float = 1e-10,
        weight_decay: float = 0.
    ):
        super(Adagrad, self).__init__(params, lr, weight_decay)
        self.eps = eps
        self.h = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        for i, (h, p) in enumerate(zip(self.h, self.params)):
            if p.requires_grad:
                # l2 penalty
                p_grad = p.grad + self.weight_decay * p.data
                # accumulate gradients
                h += p.grad ** 2
                self.h[i] = h
                # update parameters
                p.data -= self.lr * p.grad / np.sqrt(h + self.eps)
                
        super(Adagrad, self).step()
