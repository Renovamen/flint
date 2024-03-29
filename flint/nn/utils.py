import numpy as np

from ..tensor import Tensor
from .types import _tuple_2_t

def im2col(
    input: Tensor,
    kernel_size: _tuple_2_t[int],
    out_shape: _tuple_2_t[int],
    stride: _tuple_2_t[int] = (1, 1),
    dilation: _tuple_2_t[int] = (1, 1)
) -> Tensor:
    """
    Rearrange the input tensor into column vectors. This implementation is
    adopted from Stanford's CS231n assignments 2 [1].

    Parameters
    ----------
    input : Tensor of size (batch_size, in_channels, h_in, w_in)
        A padded input tensor.

    kernel_size : Tuple[int, int]
        Shape of the kernel/weights.

    out_shape : Tuple[int, int]
        Shape of the output tensor.

    stride : Tuple[int, int], optional, default=(1, 1)
        Stride of the convolution.

    dilation : Tuple[int, int], optional, default=(1, 1)
        Spacing between kernel elements.

    mode : str, optional, default='conv'
        Converting mode, 'conv' for convolution, 'pooling' for pooling.

    Return
    ------
    input_col : Tensor of size (batch_size, in_channels, h_out, w_out)
        A rearranged tensor.

    References
    ----------
    1. `CS231n Assignments 2 <https://github.com/cs231n/cs231n.github.io/tree/master/assignments/2020>`_
    """
    batch_size, in_channels, h_in, w_in = input.shape
    kernel_h, kernel_w = kernel_size
    h_out, w_out = out_shape

    # get the indices for im2col
    i0 = np.repeat(np.arange(kernel_h), kernel_w)
    i0 = np.tile(i0, in_channels) * dilation[0]
    i1 = stride[0] * np.repeat(np.arange(h_out), w_out)

    j0 = np.tile(np.arange(kernel_w), kernel_h * in_channels) * dilation[1]
    j1 = stride[1] * np.tile(np.arange(w_out), h_out)

    i = i0.reshape(-1, 1) + i1.reshape(1, -1)  # (kernel_h * kernel_w * in_channels, h_out * w_out)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)  # (kernel_h * kernel_w * in_channels, h_out * w_out)
    k = np.repeat(np.arange(in_channels), kernel_h * kernel_w).reshape(-1, 1)  # (kernel_h * kernel_w * in_channels, 1)

    # transform the input tensor
    input_col = input[:, k, i, j]  # (batch_size, kernel_h * kernel_w * in_channels, L = h_out * w_out)
    return input_col
