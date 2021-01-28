import numpy as np
from .optimizer import Optimizer

class SGD(Optimizer):
    '''
    Stochastic Gradient Descent (optionally with momentum):
        v_{t+1} = momentum * v_t + g_{t+1}
        p_{t+1} = p_t - lr * v_{t+1}

    args:
        params (iterable): an iterable of Tensor
        lr (float, optional): learning rate (default: 0.01)
        momentum (float, optional): momentum factor (default: 0)
        nesterov (bool, optional): enable Nesterov momentum or not (default: False)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    '''

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
