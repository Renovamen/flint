import numpy as np
from typing import Union, Tuple

import tinyark
from ..tensor import Tensor
from ..utils import *

# ---------------------- activators ----------------------

def relu(input: Tensor) -> Tensor:
    '''
    Compute ReLU (Rectified Linear Unit) element-wise.
    '''

    out = Tensor(
        data = np.maximum(0., input.data),
        depends_on = [input],
        requires_grad = input.requires_grad
    )

    def grad_relu():
        if input.requires_grad:
            input.grad += out.grad * ((input.data > 0) * np.ones_like(input.data))

    if out.requires_grad:
        out.grad_fn = grad_relu

    return out

def sigmoid(input: Tensor) -> Tensor:
    '''
    Compute Sigmoid element-wise:
        sigmoid(x) = \frac{1}{1 + \exp(-x)}
    '''

    ret = 1 / (1 + np.exp(-input.data))

    out = Tensor(
        data = ret,
        depends_on = [input],
        requires_grad = input.requires_grad
    )

    def grad_sigmoid():
        if input.requires_grad:
            input.grad += out.grad * out.data * (1 - out.data)

    if out.requires_grad:
        out.grad_fn = grad_sigmoid

    return out

def tanh(input: Tensor) -> Tensor:
    '''
    Compute Tanh (Hyperbolic Tangent) element-wise:
        tanh(x) = \frac{sinhx}{coshx} = \frac{\exp(x) - \exp(-x)}{\exp(x) + \exp(-x)}
    '''

    ret = np.tanh(input.data)

    out = Tensor(
        data = ret,
        depends_on = [input],
        requires_grad = input.requires_grad
    )

    def grad_tanh():
        if input.requires_grad:
            input.grad += out.grad * (1 - np.square(out.data))

    if out.requires_grad:
        out.grad_fn = grad_tanh

    return out


# ---------------------- loss functions ----------------------

def nll_loss(
    input: Tensor,
    target: Tensor,
    reduction: str = 'mean'
) -> Tensor:
    '''
    Negative Log Likelihood Loss

    NOTE: Here I apply log() on the prediction data, which is DIFFERENT FROM
          F.nll_loss() IN PYTORCH!

    args:
        input (Tensor): a 2-dim (batch_size, n_classes) tensor
        target (Tensor): a 1-dim (batch_size) tensor where each value:
            0 <= target[i] <= n_classes-1
        reduction (str, optional): 'none' / 'mean' / 'sum'
    '''

    dim = input.ndim

    if dim != 2:
        raise ValueError("Expected 2 dimensions (got {})".format(dim))

    if input.shape[0] != target.shape[0]:
        raise ValueError(
            "Expected input batch_size ({}) to match target batch_size ({}).".format(input.shape[0], target.shape[0])
        )

    batch_size = input.shape[0]
    n_classes = input.shape[1]
    delta = 1e-7  # deal with the situation that input.data = 0

    ret = - np.log(input.data[np.arange(batch_size), target.data.astype(np.int)] + delta)
    if reduction in ['sum', 'mean']:
        ret = np.sum(ret)
    if reduction == 'mean':
        ret = ret / batch_size

    out = Tensor(
        data = ret,
        depends_on = [input],
        requires_grad = input.requires_grad
    )

    def grad_nll():
        if input.requires_grad:
            p = np.clip(input.data, 1e-15, 1 - 1e-15)
            y = to_categorical(target.data, n_classes=n_classes)
            if reduction == 'mean':
                input.grad += (p - y) / batch_size  # (batch_size, n_classes)
            elif reduction == 'sum':
                input.grad += (p - y)  # (batch_size, n_classes)

    if out.requires_grad and reduction != 'none':
        out.grad_fn = grad_nll

    return out

def cross_entropy(
    input: Tensor,
    target: Tensor,
    reduction: str = 'mean'
) -> Tensor:
    '''
    Cross Entropy Loss

    NOTE: Combines softmax() and nll_loss(), which is DIFFERENT FROM
          F.cross_entropy() IN PYTORCH!

    args:
        input (Tensor): a 2-dim (batch_size, n_classes) tensor
        target (Tensor): a 1-dim (batch_size) tensor where each value:
            0 <= target[i] <= n_classes-1
        reduction (str, optional): 'none' / 'mean' / 'sum'
    '''

    after_softmax = input.softmax(axis=-1)
    out = nll_loss(after_softmax, target, reduction)

    return out

def mse_loss(
    input: Tensor,
    target: Tensor,
    reduction: str = 'mean'
) -> Tensor:
    '''
    Mean Squared Error Loss: (x - y)^2

    args:
        input (Tensor): Tensor of shape (batch_size, *)
        target (Tensor): Tensor of the same shape as input
        reduction (str, optional): 'none' / 'mean' / 'sum'
    '''

    if target.shape != input.shape:
        raise ValueError(
            "The target size ({}) is different to the input size ({}). "
            "Please ensure they have the same size.".format(target.shape, input.shape)
        )

    n = input.size

    out = (input - target) ** 2
    if reduction in ['sum', 'mean']:
        out = out.sum()
    if reduction == 'mean':
        out = out / n

    return out

def binary_cross_entropy(
    input: Tensor,
    target: Tensor,
    reduction: str = 'mean'
) -> Tensor:
    '''
    Binary Cross Entropy Loss:
        loss = - (y * log(x) + (1 - y) * log(1 - x))

    args:
        input (Tensor): Tensor of shape (batch_size, *)
        target (Tensor): Tensor of the same shape as input
        reduction (str, optional): 'none' / 'mean' / 'sum'
    '''

    if target.shape != input.shape:
        raise ValueError(
            "The target size ({}) is different to the input size ({}). "
            "Please ensure they have the same size.".format(target.shape, input.shape)
        )

    n = input.size

    out = - (target * input.log() + (-target + 1.) * (-input + 1.).log())
    if reduction in ['sum', 'mean']:
        out = out.sum()
    if reduction == 'mean':
        out = out / n

    return out

# ---------------------- pad ----------------------

def pad(input: Tensor, pad: Tuple, value: int = 0) -> Tensor:
    '''
    Pad tensor.

    args:
        input (Tensor): N-dimensional tensor
        pad (tuple):
            Padding sizes, a m-elements tuple, where m/2 <= input dimensions and
            m is even. The padding sizes are described starting from the m/2 to
            last dimension to the last dimension. That is, m/2 dimensions of
            input will be padded.
        value (int):
            Fill value for 'constant' padding (default: 0)
    '''

    n_pad_dims = int(len(pad) / 2)
    ndims = input.ndim

    no_pad_width = [(0, 0) for i in range(0, ndims - n_pad_dims)]
    pad_width = no_pad_width + [(pad[i * 2], pad[i * 2 + 1]) for i in range(0, n_pad_dims)]

    ret = np.pad(
        input.data,
        pad_width = pad_width,
        mode = 'constant',
        constant_values = value,
    )

    out = Tensor(
        data = ret,
        depends_on = [input],
        requires_grad = input.requires_grad
    )

    def unpad(x: Tensor):
        slices = [slice(p[0], None if p[1] == 0 else -p[1]) for p in pad_width]
        return x[tuple(slices)]

    def grad_pad():
        if input.requires_grad:
            input.grad += unpad(out.grad)

    if out.requires_grad:
        out.grad_fn = grad_pad

    return out

# ---------------------- linear ----------------------

def linear(input: Tensor, weight: Tensor, bias: Tensor = None):
    out = input @ weight

    if bias is not None:
        out += bias

    return out

# ---------------------- conv ----------------------

def conv2d(
    input: Tensor,
    weight: Tensor,
    bias: Tensor = None,
    stride: tuple = (1, 1),
    padding: tuple = (0, 0),
    dilation: tuple = (1, 1)
):
    '''
    Apply a 2D convolution over an input signal composed of several input
    planes:

    NOTE: Use `im2col` function to perform the convolution as a single
    matrix multiplication. For more details, see ref[1].

    args:
        input (Tensor): input tensor
        weight (Tensor): weight of the conv1d layer
        bias (Tensor, optional): bias of the conv2d layer (default: None)
        stride (tuple, optional):
            stride of the convolution (default: (1, 1))
        padding (tuple, optional):
            zero-padding added to both sides of the input (default: (0, 0))
        dilation (tuple, ptional):
            spacing between kernel elements (default: (1, 1))

    shape:
        input: (batch_size, in_channels, h_in, w_in)
        output: (batch_size, out_channels, h_out, w_out)

        where:
            h_out = (h_in + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1
            w_out = (w_in + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1

    refs:
        [1] Why GEMM is at the heart of deep learning? Pete Warden. 2015.
            Blog: https://petewarden.com/2015/04/20/why-gemm-is-at-the-heart-of-deep-learning/
    '''

    batch_size, in_channels, h_in, w_in = input.shape
    out_channels, in_channels, kernel_h, kernel_w = weight.shape

    # compute the dimensions of the convolution output
    h_out = int((h_in + 2 * padding[0] - dilation[0] * (kernel_h - 1) - 1) / stride[0] + 1)
    w_out = int((w_in + 2 * padding[1] - dilation[1] * (kernel_w - 1) - 1) / stride[1] + 1)

    # padding input tensor
    padded_data = pad(input, (0, 0, 0, 0, padding[0], padding[0], padding[1], padding[1]))

    # convert input tensor and weights/kernels into a 2D matrices
    input_col = tinyark.im2col(padded_data, weight.shape, (h_out, w_out), stride, dilation)
    weight_col = weight.view(out_channels, -1)

    out = (weight_col @ input_col).view(out_channels, h_out, w_out, batch_size).permute(3, 0, 1, 2)

    if bias is not None:
        out += bias

    return out


def conv1d(
    input: Tensor,
    weight: Tensor,
    bias: Tensor = None,
    stride: tuple = (1, ),
    padding: tuple = (0, ),
    dilation: tuple = (1, )
):
    '''
    Apply a 1D convolution over an input signal composed of several input
    planes:

    args:
        input (Tensor): input tensor
        weight (Tensor): weight of the conv1d layer
        bias (Tensor, optional): bias of the conv1d layer
        stride (tuple[int], optional):
            stride of the convolution (default: (1))
        padding (tuple[int], optional):
            zero-padding added to both sides of the input (default: (0))
        dilation (tuple[int], optional):
            spacing between kernel elements (default: (1))

    shape:
        input: (batch_size, in_channels, L_in)
        weight: (out_channels, in_channels, kernel_size)
        bias: (1, out_channels, 1)
        output: (batch_size, out_channels, L_out)

        where:
            L_out = (L_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
    '''

    # add a dimension to tensors so we can use conv2d
    input_2d = input.unsqueeze(dim=2)
    weight_2d = weight.unsqueeze(dim=2)
    bias_2d = bias.unsqueeze(dim=2)

    stride_2d = (1, stride[0])
    pad_2d = (0, padding[0])
    dilation_2d = (1, dilation[0])

    out_2d = conv2d(input_2d, weight_2d, bias_2d, stride_2d, pad_2d, dilation_2d)  # (batch_size, out_channels, 1, L_out)

    # drop the added dimension
    out = out_2d.squeeze(dim=2)
    return out
