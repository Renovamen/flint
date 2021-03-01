import numpy as np
from .optimizer import Optimizer

class Adagrad(Optimizer):
    """
    Implementation of Adagrad algorithm proposed in [1].

    .. math::
      h_t = h_{t-1} + g_t^2
    .. math::
      \\theta_{t+1} = \\theta_t - \\frac{\\text{lr}}{\sqrt{h_t + \epsilon}} \cdot g_t

    Args:
        params (iterable): An iterable of Tensor
        lr (float, optional, default=0.01): Learning rate
        eps (float, optional, default=1e-10): Term added to the
            denominator to improve numerical stability
        weight_decay (float, optional, default=0): weight decay (L2 penalty)

    References
    ----------
    1. "`Adaptive Subgradient Methods for Online Learning and Stochastic Optimization. <https://jmlr.org/papers/volume12/duchi11a/duchi11a.pdf>`_" John Duchi, et al. JMRL 2011.
    """

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
                # accumulate squared gradients
                h += p.grad ** 2
                self.h[i] = h
                # update parameters
                p.data -= self.lr * p.grad / np.sqrt(h + self.eps)

        super(Adagrad, self).step()
