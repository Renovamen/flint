import numpy as np
from .optimizer import Optimizer

class SGD(Optimizer):
    """
    Implementation of Stochastic Gradient Descent (optionally with
    momentum).

    .. math::
       v_{t+1} = \mu \cdot v_t + g_{t+1}

    .. math::
       \\theta_{t+1} = \\theta_t - \\text{lr} \cdot v_{t+1}

    where :math:`\\theta`, :math:`g`, :math:`v` and :math:`\mu` denote the
    parameters, gradient, velocity, and momentum respectively.

    Args:
        params (iterable): an iterable of Tensor
        lr (float, optional): learning rate (default: 0.01)
        momentum (float, optional): momentum factor (default: 0)
        nesterov (bool, optional): enable Nesterov momentum or not (default: False)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    """

    def __init__(
        self,
        params = None,
        lr: float = 0.01,
        momentum: float = 0.,
        nesterov: bool = False,
        weight_decay: float = 0.
    ):
        super(SGD, self).__init__(params, lr, weight_decay)
        self.momentum = momentum
        self.nesterov = nesterov
        self.v = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        for i, (v, p) in enumerate(zip(self.v, self.params)):
            if p.requires_grad:
                # l2 penalty
                p_grad = p.grad + self.weight_decay * p.data
                # heavy ball / polyak's momentum
                v = self.momentum * v + p_grad
                self.v[i] = v
                # nesterov's momentum
                if self.nesterov:
                    v = self.momentum * v + p_grad
                # update parameters
                p.data -= self.lr * v

        super(SGD, self).step()
