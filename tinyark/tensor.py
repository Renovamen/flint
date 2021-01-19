# inspired by https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py

import numpy as np
from typing import Union

Arrayable = Union[float, list, np.ndarray]

def ensure_ndarray(data: Arrayable) -> np.ndarray:
    if isinstance(data, np.ndarray):
        return data
    else:
        return np.asarray(data)

class Tensor(object):
    '''
    Tensor is the basic structure in the computation graph. It holds value
    for forward computation and grad for backward propagation.

    args:
        data (float / list / ndarray): data for the Tensor
        depends_on (list): list of dependent tensors (used when building autograd graph)
        requires_grad (bool): if the Tensor requires gradient
    '''

    def __init__(
        self,
        data: Arrayable,
        depends_on: list = [],
        requires_grad: bool = False
    ) -> None:
        self.data = ensure_ndarray(data)
        self.requires_grad = requires_grad
        self.grad: Optional[np.ndarray] = None
        self.grad_fn = None  # a function for computing gradients
        self.depends_on = []
        self.add_depends_on(depends_on)

        if self.requires_grad:
            self.zero_grad()

    def zero_grad(self) -> None:
        '''
        Set the gradient to zero.
        '''
        self.grad = np.zeros_like(self.data, dtype=np.float32)
    
    def add_depends_on(self, depends_on: list = []) -> None:
        '''
        Add the dependent tensors for building autograd graph.

        args:
            depends_on (list): list of dependent tensors
        '''
        for i in depends_on:
            if isinstance(i, Tensor):
                self.depends_on.append(i)
            else:
                raise TypeError('Expected Tensor but got %s' % type(i))
    
    def backward(self):
        '''
        Autograd on computation graph.
        '''
        if self.grad_fn is None:
            raise ValueError('Can not solve grad on %s' % self)

        # build autograd graph
        graph = []
        visited = set()

        def dfs(v):
            if v not in visited:
                visited.add(v)
                for prev in v.depends_on:
                    dfs(prev)
                graph.append(v)
        
        dfs(self)
        
        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1.
        for node in reversed(graph):
            if node.grad_fn is not None:
                node.grad_fn()

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

    # -------------- overwrite operators --------------

    def __add__(self, other: 'Tensor') -> 'Tensor':
        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(
            data = self.data + other.data,
            depends_on = [self, other],
            requires_grad = self.requires_grad or other.requires_grad
        )

        def grad_add():
            if self.requires_grad:
                self.grad += out.grad
            if other.requires_grad:
                other.grad += out.grad

        if out.requires_grad:
            out.grad_fn = grad_add

        return out

    def __radd__(self, other: 'Tensor') -> 'Tensor':
        return self.__add__(other)

    def __sub__(self, other: 'Tensor') -> 'Tensor':
        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(
            data = self.data - other.data,
            depends_on = [self, other],
            requires_grad = self.requires_grad or other.requires_grad
        )

        def grad_sub():
            if self.requires_grad:
                self.grad += out.grad
            if other.requires_grad:
                other.grad -= out.grad

        if out.requires_grad:
            out.grad_fn = grad_sub

        return out

    def __rsub__(self, other: 'Tensor') -> 'Tensor':
        return self.__sub__(other)

    def __mul__(self, other: 'Tensor') -> 'Tensor':
        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(
            data = self.data * other.data,
            depends_on = [self, other],
            requires_grad = self.requires_grad or other.requires_grad
        )

        def grad_mul():
            if self.requires_grad:
                self.grad += out.grad * other.data
            if other.requires_grad:
                other.grad += out.grad * self.data

        if out.requires_grad:
            out.grad_fn = grad_mul

        return out

    def __rmul__(self, other: 'Tensor') -> 'Tensor':
        return self.__mul__(other)
    
    def __truediv__(self, other: 'Tensor') -> 'Tensor':
        '''
        c = a / b
        dc/da = 1 / b, dc/db = - (a / b^2)
        '''

        print(other)

        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(
            data = self.data / other.data,
            depends_on = [self, other],
            requires_grad = self.requires_grad or other.requires_grad
        )

        def grad_div():
            if self.requires_grad:
                self.grad += out.grad / other.data
            if other.requires_grad:
                other.grad += - (out.grad * self.data / (other.data ** 2))

        if out.requires_grad:
            out.grad_fn = grad_div

        return out
    
    def __rtruediv__(self, other: 'Tensor') -> 'Tensor':
        return self.__truediv__(other)

    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(
            data = np.dot(self.data, other.data),
            depends_on = [self, other],
            requires_grad = self.requires_grad or other.requires_grad
        )

        def grad_mm():
            if self.requires_grad:
                self.grad += np.dot(out.grad, other.data.T)
            if other.requires_grad:
                other.grad += np.dot(self.data.T, out.grad)

        if out.requires_grad:
            out.grad_fn = grad_mm

        return out

    def __rmatmul__(self, other: 'Tensor') -> 'Tensor':
        return self.__matmul__(other)
    
    def __pow__(self, exp: Union[int, float]) -> 'Tensor':
        out = Tensor(
            data = self.data ** exp,
            depends_on = [self],
            requires_grad = self.requires_grad
        )

        def grad_pow():
            if self.requires_grad:
                self.grad += (exp * self.data ** (exp - 1)) * out.grad

        if out.requires_grad:
            out.grad_fn = grad_pow

        return out

    def __rpow__(self, exp: Union[int, float]) -> 'Tensor':
        return self.__pow__(exp)

    def __neg__(self) -> 'Tensor':
        out = Tensor(
            data = -self.data,
            depends_on = [self],
            requires_grad = self.requires_grad
        )

        def grad_neg():
            if self.requires_grad:
                self.grad = -out.grad

        if out.requires_grad:
            out.grad_fn = grad_neg

        return out
