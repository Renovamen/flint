import numpy as np
from .optimizer import Optimizer

class Adadelta(Optimizer):
    '''
    Adadelta:
        h_t = \gamma * h_{t-1} + (1 - \gamma) * (g_t)^2
        g'_t = sqrt((delta_{t-1} + eps) / (h_t + eps)) * g_t
        delta_t = rho * delta_{t-1} + (1 - rho) * (g'_t)^2
        p_{t+1} = p_t - g'_t

    args:
        params (iterable): an iterable of Tensor
        rho (float, optional): coefficient used for computing a running average
            of squared gradients (default: 0.9)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-6)
        lr (float, optional): coefficient that scale delta before it is applied
            to the parameters (default: 1.0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    
    refs:
        ADADELTA: An Adaptive Learning Rate Method. Matthew D. Zeiler. arxiv 2012.
        Paper: https://arxiv.org/abs/1212.5701
    '''

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
