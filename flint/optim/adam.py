import numpy as np
from typing import Tuple
from .optimizer import Optimizer

class Adam(Optimizer):
    """
    Implementation of Adam algorithm proposed in [1].

    .. math::
        v_t = \\beta_1 v_{t-1} + (1 - \\beta_1) g_t
    .. math::
        h_t = \\beta_2 h_{t-1} + (1 - \\beta_2) g_t^2

    Bias correction:

    .. math::
        \hat{v}_t = \\frac{v_t}{1 - \\beta_1^t}
    .. math::
        \hat{h}_t = \\frac{h_t}{1 - \\beta_2^t}

    Update parameters:

    .. math::
        \\theta_t = \\theta_{t-1} - \\text{lr} \cdot \\frac{\hat{v}_t}{\sqrt{\hat{h}_t + \epsilon}}

    Args:
        params (iterable): An iterable of Tensor
        lr (float, optional, default=1e-3): Learning rate
        betas (Tuple[float, float], optional, default=(0.9, 0.999)):
            Coefficients used for computing running averages of gradient
            and its square
        eps (float, optional, default=1e-8): Term added to the denominator
            to improve numerical stability
        weight_decay (float, optional, default=0): Weight decay (L2 penalty)

    References
    ----------
    1. "`Adam: A Method for Stochastic Optimization. <https://arxiv.org/abs/1412.6980>`_" Diederik P. Kingma and Jimmy Ba. ICLR 2015.
    """

    def __init__(
        self,
        params = None,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.
    ):
        super(Adam, self).__init__(params, lr, weight_decay)
        self.eps = eps
        self.beta1, self.beta2 = betas
        self.v = [np.zeros_like(p.data) for p in self.params]
        self.h = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        super(Adam, self).step()

        for i, (v, h, p) in enumerate(zip(self.v, self.h, self.params)):
            if p.requires_grad:
                # l2 penalty
                p_grad = p.grad + self.weight_decay * p.data
                # moving average of gradients
                v = self.beta1 * v + (1 - self.beta1) * p.grad
                self.v[i] = v
                # moving average of squared gradients
                h = self.beta2 * h + (1 - self.beta2) * (p.grad ** 2)
                self.h[i] = h
                # bias correction
                v_correction = 1 - (self.beta1 ** self.iterations)
                h_correction = 1 - (self.beta2 ** self.iterations)
                # update parameters
                p.data -= (self.lr / v_correction * v) / (np.sqrt(h) / np.sqrt(h_correction) + self.eps)
