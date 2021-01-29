import numpy as np
from .optimizer import Optimizer

class RMSprop(Optimizer):
    '''
    RMSprop:
        h_t = gamma * h_{t-1} + (1 - gamma) * (g_t)^2
        p_{t+1} = p_t - lr / sqrt(h_t + eps) * g_t

    args:
        params (iterable): an iterable of Tensor
        lr (float, optional): learning rate (default: 0.01)
        alpha (float, optional): coefficient used for computing a running average
            of squared gradients  (default: 0.99)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    
    refs:
        Neural Networks for Machine	Learning, Lecture 6e - rmsprop: Divide the
        gradient by a running average of its recent magnitude. Geoffrey	Hinton.
        Course Slide: https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
    '''

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
