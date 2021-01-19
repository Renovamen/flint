import numpy as np

class Tensor:
    '''
    Tensor is the basic structure in the computation graph. It holds value
    for forward computation and grad for backward propagation.

    args:
        data (ndarray): data for the Tensor
        requires_grad (bool): if the Tensor requires gradient
    '''

    def __init__(
        self,
        data: np.ndarray,
        requires_grad: bool = False
    ):
        self.data = data
        self.requires_grad = requires_grad
        self.grad = Optional[np.ndarray] = None

        if self.requires_grad:
            self.zero_grad()

    def zero_grad(self) -> None:
        '''
        Set the gradient to zero.
        '''
        self.grad = np.zeros_like(self.data, dtype=np.float32)
    
    @classmethod
    def zeros(cls, *shape, **kwargs):
        return cls(np.zeros(shape, dtype=np.float32), **kwargs)

    @classmethod
    def ones(cls, *shape, **kwargs):
        return cls(np.ones(shape, dtype=np.float32), **kwargs)

    @classmethod
    def randn(cls, *shape, **kwargs):
        return cls(np.random.randn(*shape).astype(np.float32), **kwargs)

    @property
    def shape(self):
        return self.data.shape
    
    @property
    def dtype(self):
        return self.data.dtype
    
    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @data.setter
    def assign(self, x: np.ndarray) -> None:
        self.data = x
        # setting the data manually means we invalidate the gradient
        self.grad = None