import numpy as np
from typing import Tuple
from .tensor import Tensor

def eq(input: Tensor, target: Tensor) -> Tensor:
    if input.shape != target.shape:
        raise ValueError(
            "Expected input shape ({}) to match target shape ({}).".format(input.shape, target.shape)
        )
    return Tensor(np.equal(input.data, target.data))

def im2col(
    input: Tensor,
    kernel_shape: Tuple,
    out_shape: Tuple,
    stride: tuple = (1, 1),
    dilation: tuple = (1, 1)
) -> Tensor:
    """
    Rearrange the input tensor into column vectors. This implementation is
    adopted from Stanford's CS231n assignments 2 [1].

    Args:
        input (Tensor): A padded input tensor
        kernel_shape (tuple): Shape of the kernel/weights
        out_shape (tuple): Shape of the output tensor
        stride (tuple, optional, default=(1, 1)): Stride of the convolution
        dilation (tuple, optional, default=(1, 1)): Spacing between kernel elements

    References
    ----------
    1. `CS231n Assignments 2 <https://github.com/cs231n/cs231n.github.io/tree/master/assignments/2020>`_
    """
    batch_size, in_channels, h_in, w_in = input.shape
    out_channels, in_channels, kernel_h, kernel_w = kernel_shape
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
    input_col = input[:, k, i, j]
    input_col = input_col.permute(1, 2, 0).view(kernel_h * kernel_w * in_channels, -1)

    return input_col
