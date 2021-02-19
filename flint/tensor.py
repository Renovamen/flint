# inspired by https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py

import numpy as np
from typing import Union
from .utils import *

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
        Fill the gradient with zeros.
        '''
        self.grad = np.zeros(self.shape, dtype=np.float32)

    def one_grad(self) -> None:
        '''
        Fill the gradient with ones.
        '''
        self.grad = np.ones(self.shape, dtype=np.float32)

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
        self.one_grad()
        for node in reversed(graph):
            if node.grad_fn is not None:
                node.grad_fn()

    @classmethod
    def zeros(cls, *shape, **kwargs):
        '''
        Create a tensor filled with the scalar value `0`.
        '''
        return cls(np.zeros(shape, dtype=np.float32), **kwargs)

    @classmethod
    def ones(cls, *shape, **kwargs):
        '''
        Create a tensor filled with the scalar value `1`.
        '''
        return cls(np.ones(shape, dtype=np.float32), **kwargs)

    @classmethod
    def randn(cls, *shape, **kwargs):
        '''
        Create a tensor filled with random scalar values.
        '''
        return cls(np.random.randn(*shape).astype(np.float32), **kwargs)

    # -------------- properties --------------

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

    # -------------- operator overloading --------------

    def __add__(self, other: 'Tensor') -> 'Tensor':
        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(
            data = self.data + other.data,
            depends_on = [self, other],
            requires_grad = self.requires_grad or other.requires_grad
        )

        def grad_add():
            if self.requires_grad:
                # self.grad += out.grad
                self.grad = broadcast_add(self.grad, out.grad)
            if other.requires_grad:
                # other.grad += out.grad
                other.grad = broadcast_add(other.grad, out.grad)

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
                # self.grad += out.grad
                self.grad = broadcast_add(self.grad, out.grad)
            if other.requires_grad:
                # other.grad -= out.grad
                other.grad = broadcast_add(other.grad, -out.grad)

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
                # self.grad += out.grad * other.data
                self.grad = broadcast_add(self.grad, out.grad * other.data)
            if other.requires_grad:
                # other.grad += out.grad * self.data
                other.grad = broadcast_add(other.grad, out.grad * self.data)

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

        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(
            data = self.data / other.data,
            depends_on = [self, other],
            requires_grad = self.requires_grad or other.requires_grad
        )

        def grad_div():
            if self.requires_grad:
                # self.grad += out.grad / other.data
                self.grad = broadcast_add(self.grad, out.grad / other.data)
            if other.requires_grad:
                # other.grad += - (out.grad * self.data / (other.data ** 2))
                other.grad = broadcast_add(other.grad, - (out.grad * self.data / (other.data ** 2)))
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
                self.grad += -out.grad

        if out.requires_grad:
            out.grad_fn = grad_neg

        return out

    # -------------- other maths --------------

    def exp(self) -> 'Tensor':
        out = Tensor(
            data = np.exp(self.data),
            depends_on = [self],
            requires_grad = self.requires_grad
        )

        def grad_exp():
            if self.requires_grad:
                self.grad += out.grad * out.data

        if out.requires_grad:
            out.grad_fn = grad_exp

        return out

    def log(self)  -> 'Tensor':
        out = Tensor(
            data = np.log(self.data),
            depends_on = [self],
            requires_grad = self.requires_grad
        )

        def grad_log():
            if self.requires_grad:
                self.grad += out.grad / self.data

        if out.requires_grad:
            out.grad_fn = grad_log

        return out

    def sum(self, axis: int = None, keepdims: bool = False) -> 'Tensor':
        out = Tensor(
            data = np.sum(self.data, axis=axis, keepdims=keepdims),
            depends_on = [self],
            requires_grad = self.requires_grad
        )

        def grad_sum():
            if self.requires_grad:
                out_grad = out.grad
                if out.ndim < self.ndim:
                    sum_axis = [axis] if type(axis) is int else axis
                    expanded_shape = [1 if sum_axis is None or i in sum_axis else self.shape[i] for i in range(len(self.shape))]
                    out_grad = out_grad.reshape(expanded_shape)
                self.grad += out_grad + np.zeros_like(self.data)

        if out.requires_grad:
            out.grad_fn = grad_sum

        return out

    def max(self, axis: int = None, keepdims: bool = False) -> 'Tensor':
        '''
        Return the maximum value of all elements in the tensor.
        '''

        out = Tensor(
            data = np.max(self.data, axis=axis, keepdims=keepdims),
            depends_on = [self],
            requires_grad = self.requires_grad
        )

        def grad_max():
            if self.requires_grad:
                out_grad = out.grad
                out_data = out.data
                if out.ndim < self.ndim:
                    max_axis = [axis] if type(axis) is int else axis
                    # here I don't use np.expand_dims(), because I have to deal
                    # with the situation when axis = None
                    expanded_shape = [1 if max_axis is None or i in max_axis else self.shape[i] for i in range(len(self.shape))]
                    out_grad = out_grad.reshape(expanded_shape)
                    out_data = out_data.reshape(expanded_shape)
                mask = (self.data == out_data)
                self.grad += mask * out_grad

        if out.requires_grad:
            out.grad_fn = grad_max

        return out

    def argmax(self, axis: int = None) -> 'Tensor':
        '''
        Return the indice of the maximum value of all elements in the tensor.
        '''
        out = Tensor(np.argmax(self.data, axis=axis))
        return out

    def softmax(self, axis: int = -1) -> 'Tensor':
        out = self - self.max(axis=axis, keepdims=True)
        out = out.exp()
        out = out / out.sum(axis=axis, keepdims=True)
        return out

    def log_softmax(self, axis: int = -1) -> 'Tensor':
        after_softmax = self.softmax(axis)
        out = after_softmax.log()
        return out

    # -------------- movement operations --------------

    def __getitem__(self, item):
        out = Tensor(
            data = self.data[item],
            depends_on = [self],
            requires_grad=self.requires_grad
        )

        def grad_slice():
            if self.requires_grad:
                self.grad[item] += out.grad

        if out.requires_grad:
            out.grad_fn = grad_slice

        return out

    def view(self, *shape) -> 'Tensor':
        '''
        Returns a new tensor with the same data as the self tensor but of
        a different shape.

        args:
            *shape: the desired size
        '''

        out = Tensor(
            data = np.reshape(self.data, shape),
            depends_on = [self],
            requires_grad = self.requires_grad
        )

        def grad_view():
            self.grad += np.reshape(out.grad, self.shape)

        if out.requires_grad:
            out.grad_fn = grad_view

        return out

    def permute(self, *dims) -> 'Tensor':
        '''
        Returns a view of the original tensor with its dimensions permuted.

        args:
            *dims: the desired ordering of dimensions
        '''

        out = Tensor(
            data = self.data.transpose(dims),
            depends_on = [self],
            requires_grad = self.requires_grad
        )

        def grad_permute():
            self.grad += out.grad.transpose(np.argsort(dims))

        if out.requires_grad:
            out.grad_fn = grad_permute

        return out

    def transpose(self, dim0: int, dim1: int) -> 'Tensor':
        '''
        Swap the dimension dim0 and dim1 of the tensor.

        args:
            dim0: the first dimension to be transposed
            dim1: the second dimension to be transposed
        '''

        def get_dim(dim):
            if dim == dim0:
                return dim1
            elif dim == dim1:
                return dim0
            else:
                return dim

        dims = tuple([get_dim(i) for i in range(self.ndim)])

        out = Tensor(
            data = self.data.transpose(dims),
            depends_on = [self],
            requires_grad = self.requires_grad
        )

        def grad_transpose():
            self.grad += out.grad.transpose(np.argsort(dims))

        if out.requires_grad:
            out.grad_fn = grad_transpose

        return out

    def unsqueeze(self, dim: int) -> 'Tensor':
        '''
        Insert a dimension of size one at the specified position.

        args:
            dim (int): the index at which to insert the singleton dimension
        '''

        out = Tensor(
            data = np.expand_dims(self.data, axis=dim),
            depends_on = [self],
            requires_grad = self.requires_grad
        )

        def grad_unsqueeze():
            self.grad = broadcast_add(self.grad, np.squeeze(out.grad, axis=dim))

        if out.requires_grad:
            out.grad_fn = grad_unsqueeze

        return out

    def squeeze(self, dim: int = None) -> 'Tensor':
        '''
        Remove the dimensions of input of size 1.

        args:
            dim (int, optional):
                If given, the input will be squeezed only in this dimension. Or
                all the dimensions of size 1 will be removed.
        '''

        out = Tensor(
            data = np.squeeze(self.data, axis=dim),
            depends_on = [self],
            requires_grad = self.requires_grad
        )

        def grad_squeeze():
            self.grad += np.reshape(out.grad, self.shape)

        if out.requires_grad:
            out.grad_fn = grad_squeeze

        return out

    # -------------- initializing --------------

    def fill_(self, val: float) -> None:
        '''
        Fill the tensor with the given scalar value `val`.

        args:
            val (float): the value to fill the tensor with
        '''
        self.data.fill(val)

    def zero_(self) -> None:
        '''
        Fill the tensor with the scalar value `0`.
        '''
        self.fill_(0.)

    def one_(self) -> None:
        '''
        Fill the tensor with the scalar value `1`.
        '''
        self.fill_(1.)

    def uniform_(self, low: float = 0., high: float = 1.) -> None:
        '''
        Fill the tensor with values drawn from the uniform distribution.

        args:
            low (float): the lower bound of the uniform distribution
            high (float): the upper bound of the uniform distribution
        '''
        self.data = np.random.uniform(low=low, high=high, size=self.shape)

    def normal_(self, mean: float = 0., std: float = 1.) -> None:
        '''
        Fill the tensor with values drawn from the normal distribution.

        args:
            mean (float): the mean of the normal distribution
            std (float): the standard deviation of the normal distribution
        '''
        self.data = np.random.normal(loc=mean, scale=std, size=self.shape)