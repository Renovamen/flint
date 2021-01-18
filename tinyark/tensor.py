import numpy as np

class Tensor:
    '''
    Tensor is the basic structure in the computation graph. It holds value
    for forward computation and grad for backward propagation.

    args:
        name (str): name for the Tensor
        shape (Tuple): shape for the Tensor
        requires_grad (bool): if the Tensor requires gradient
        dtype: numpy data type for the Tensor
    '''

    def __init__(self, name: str = None, shape: Tuple = (1, ), requires_grad: bool = False, dtype = np.float32):
        self.dtype = dtype
        self.data = np.zeros(shape = shape, dtype = dtype)
        self.grad = np.zeros(shape, dtype = np.float32)
        self.requires_grad = requires_grad

    def zero_grad(self):
        '''
        Set the gradient to zero.
        '''
        self.grad.zero_()
    
    @property
    def shape(self):
        return self.data.shape
    
    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    
    
