import numpy as np
from .optimizer import Optimizer

class RMSprop(Optimizer):
    """
    Implementation of RMSprop algorithm proposed in [1].

    .. math::
        h_t = \\alpha h_{t-1} + (1 - \\alpha) g_t^2
    .. math::
        \\theta_{t+1} = \\theta_t - \\frac{\\text{lr}}{\sqrt{h_t + \epsilon}} \cdot g_t

    Args:
        params (iterable): An iterable of Tensor
        lr (float, optional, default=0.01): Learning rate
        alpha (float, optional, default=0.99): Coefficient used for
            computing a running average of squared gradients
        eps (float, optional, default=1e-8): Term added to the denominator
            to improve numerical stability
        weight_decay (float, optional, default=0): Weight decay (L2 penalty)

    References
    ----------
    1. "`Neural Networks for Machine Learning, Lecture 6e - rmsprop: Divide the gradient by a running average of its recent magnitude. <https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf>`_" Geoffrey Hinton.
    """

    def __init__(
        self,
        params = None,
        lr: float = 0.01,
        alpha: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0.
    ):
        super(RMSprop, self).__init__(params, lr, weight_decay)
        self.eps = eps
        self.alpha = alpha
        self.h = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        for i, (h, p) in enumerate(zip(self.h, self.params)):
            if p.requires_grad:
                # l2 penalty
                p_grad = p.grad + self.weight_decay * p.data
                # moving average of the squared gradients
                h = self.alpha * h + (1 - self.alpha) * (p.grad ** 2)
                self.h[i] = h
                # update parameters
                p.data -= self.lr * p.grad / np.sqrt(h + self.eps)

        super(RMSprop, self).step()
