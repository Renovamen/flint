class Optimizer:
    '''
    Base class for all optimizers.
    
    args:
        params (iterable): an iterable of Tensor
        lr (float): learning rate
        weight_decay (float): weight decay (L2 penalty)
    '''

    def __init__(self, params = None, lr: float = 0.01, weight_decay: float = 0.):
        self.params = params
        self.lr = lr
        self.weight_decay = weight_decay
        self.iterations = 0

    def zero_grad(self):
        '''
        Set the gradients of all parameters to zero.
        '''
        for p in self.params:
            p.zero_grad()

    def step(self):
        self.iterations += 1
