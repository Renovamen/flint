import numpy as np
from .optimizer import Optimizer

class Adadelta(Optimizer):
    """
    Implementation of Adadelta algorithm proposed in [1].

    .. math::
       h_t = \\rho h_{t-1} + (1 - \\rho) g_t^2
    .. math::
       g'_t = \sqrt{\\frac{\Delta \\theta_{t-1} + \epsilon}{h_t + \epsilon}} \cdot g_t
    .. math::
       \Delta \\theta_t = \\rho \Delta \\theta_{t-1} + (1 - \\rho) (g'_t)^2
    .. math::
       \\theta_t = \\theta_{t-1} - g'_t

    where :math:`h` is the moving average of the squared gradients,
    :math:`\epsilon` is for improving numerical stability.

    Parameters
    ----------
    params : iterable
        An iterable of Tensor

    rho : float, optional, default=0.9
        Coefficient used for computing a running average of squared gradients

    eps : float, optional, default=1e-6
        Term added to the denominator to improve numerical stability

    lr : float, optional, default=1.0
        Coefficient that scale delta before it is applied to the parameters

    weight_decay : float, optional, default=0
        Weight decay (L2 penalty)

    References
    ----------
    1. "`ADADELTA: An Adaptive Learning Rate Method. Matthew D. Zeiler. <https://arxiv.org/abs/1212.5701>`_" arxiv 2012.
    """

    def __init__(
        self,
        params = None,
        rho: float = 0.99,
        eps: float = 1e-6,
        lr: float = 1.0,
        weight_decay: float = 0.
    ):
        super(Adadelta, self).__init__(params, lr, weight_decay)
        self.eps = eps
        self.rho = rho
        self.h = [np.zeros_like(p.data) for p in self.params]
        self.delta = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        for i, (h, delta, p) in enumerate(zip(self.h, self.delta, self.params)):
            if p.requires_grad:
                # l2 penalty
                p_grad = p.grad + self.weight_decay * p.data
                # moving average of the squared gradients
                h = self.rho * h + (1 - self.rho) * (p.grad ** 2)
                self.h[i] = h
                # compute g'_t and delta_t
                g_ = np.sqrt(delta + self.eps) / np.sqrt(h + self.eps) * p.grad
                delta = self.rho * delta + (1 - self.rho) * (g_ ** 2)
                self.delta[i] = delta
                # update parameters
                p.data -= self.lr * g_

        super(Adadelta, self).step()
