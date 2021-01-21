from .optimizer import Optimizer

class SGD(Optimizer):
    '''
    Implements stochastic gradient descent.
    
    args:
        params (iterable): an iterable of Tensor
        lr (float): learning rate
        weight_decay (float): weight decay (L2 penalty)
    '''

    def __init__(self, params = None, lr: float = 0.01, weight_decay: float = 0.):
        super().__init__(params, lr, weight_decay)

    def step(self):
        for p in self.params:
            if p.requires_grad:
                p.data = p.data * (1.0 - self.weight_decay) - self.lr * p.grad
        super(SGD, self).step()
