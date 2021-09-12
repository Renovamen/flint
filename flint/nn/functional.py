import math
import numpy as np
from typing import Union, Tuple

import flint
from ..tensor import Tensor
from ..utils import *
from .types import _tuple_1_t, _tuple_2_t, _tuple_any_t, _size_2_t
from .utils import im2col
from .modules.utils import _pair

# ---------------------- activators ----------------------

def relu(input: Tensor) -> Tensor:
    """
    Compute ReLU (Rectified Linear Unit) element-wise.
    """
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

def leaky_relu(input: Tensor, negative_slope: float = 0.01) -> Tensor:
    """
    Compute Leaky ReLU element-wise.

    .. math::
        \\text{LeakyReLU}(x) = \max(0, x) + \\text{negative\_slope} * \min(0, x)

    Parameters
    ----------
    negative_slope : float, optional, default=1e-2
        Controls the angle of the negative slope.
    """
    out = Tensor(
        data = np.maximum(negative_slope * input.data, input.data),
        depends_on = [input],
        requires_grad = input.requires_grad
    )

    def grad_leaky_relu():
        if input.requires_grad:
            grad = np.ones_like(input.data)
            grad[input.data < 0] = negative_slope
            input.grad += out.grad * grad

    if out.requires_grad:
        out.grad_fn = grad_leaky_relu

    return out

def sigmoid(input: Tensor) -> Tensor:
    """
    Compute Sigmoid element-wise.

    .. math::
        \\text{sigmoid}(x) = \\frac{1}{1 + \exp(-x)}
    """
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
    """
    Compute Tanh (Hyperbolic Tangent) element-wise.

    .. math::
        \\text{tanh}(x) = \\frac{\sinh(x)}{\cosh(x)} = \\frac{\exp(x) - \exp(-x)}{\exp(x) + \exp(-x)}
    """
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

def gelu(input: Tensor) -> Tensor:
    """
    Compute GELU (Gaussian Error Linear Units) [1] element-wise.

    .. math::
        \\text{GELU}(x) = x \cdot \Phi(x) = x \cdot \\frac{1}{2} [1 + \\text{erf} (x / \sqrt{2})]

    where :math:`\Phi(x)` is the Cumulative Distribution Function for Gaussian Distribution.

    We can approximate it with:

    .. math::
        \\text{GELU}(x) = 0.5 x (1 + \\text{tanh}[ \sqrt{2 / \pi} (x + 0.044715 x^3) ])

    or

    .. math::
        \\text{GELU}(x) = x \sigma(1.702 x)

    References
    ----------
    1. "`Gaussian Error Linear Units (GELUs). <https://arxiv.org/pdf/1606.08415.pdf>`_" \
        Dan Hendrycks and Kevin Gimpel. arXiv 2016.
    """
    out = 0.5 * input * (1.0 + tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * (input ** 3.0))))
    return out


# ---------------------- loss functions ----------------------

def nll_loss(
    input: Tensor,
    target: Tensor,
    reduction: str = 'mean'
) -> Tensor:
    """
    Negative Log Likelihood Loss

    NOTE:
        Here I apply ``log()`` on the prediction data, which is DIFFERENT
        FROM ``nn.functional.nll_loss()`` IN PYTORCH!

    Parameters
    ----------
    input : Tensor
        A 2-dim (batch_size, n_classes) tensor

    target : Tensor
        A 1-dim (batch_size) tensor where each value: 0 <= target[i] <= n_classes-1

    reduction : str, optional, default='mean'
        'none' / 'mean' / 'sum'
    """
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
    """
    Cross Entropy Loss

    NOTE:
        Combine ``softmax()`` and ``nll_loss()``, which is DIFFERENT FROM
        ``nn.functional.cross_entropy()`` IN PYTORCH!

    Parameters
    ----------
    input : Tensor
        A 2-dim (batch_size, n_classes) tensor

    target : Tensor
        A 1-dim (batch_size) tensor where each value: 0 <= target[i] <= n_classes-1

    reduction : str, optional, default='mean'
        'none' / 'mean' / 'sum'
    """
    after_softmax = input.softmax(dim=-1)
    out = nll_loss(after_softmax, target, reduction)

    return out

def mse_loss(
    input: Tensor,
    target: Tensor,
    reduction: str = 'mean'
) -> Tensor:
    """
    Mean Squared Error Loss :math:`(x - y)^2`

    Parameters
    ----------
    input : Tensor
        Tensor of shape (batch_size, *)

    target : Tensor
        Tensor of the same shape as input

    reduction : str, optional, default='mean'
        'none' / 'mean' / 'sum'
    """
    if target.shape != input.shape:
        raise ValueError(
            "The target size ({}) is different to the input size ({}). "
            "Please ensure they have the same size.".format(target.shape, input.shape)
        )

    n = input.numel

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
    """
    Binary Cross Entropy Loss

    .. math::
        \\text{loss} = - (y \log(x) + (1 - y) \log(1 - x))

    Parameters
    ----------
    input : Tensor
        Tensor of shape (batch_size, *)

    target : Tensor
        Tensor of the same shape as input

    reduction : str, optional, default='mean'
        'none' / 'mean' / 'sum'
    """
    if target.shape != input.shape:
        raise ValueError(
            "The target size ({}) is different to the input size ({}). "
            "Please ensure they have the same size.".format(target.shape, input.shape)
        )

    n = input.numel

    out = - (target * input.log() + (-target + 1.) * (-input + 1.).log())
    if reduction in ['sum', 'mean']:
        out = out.sum()
    if reduction == 'mean':
        out = out / n

    return out

# ---------------------- pad ----------------------

def pad(input: Tensor, pad: _tuple_any_t[int], value: int = 0) -> Tensor:
    """
    Pad tensor.

    Parameters
    ----------
    input : Tensor
        N-dimensional tensor

    pad : _tuple_any_t[int]
        Padding sizes, a m-elements tuple, where ``m/2`` <= input dimensions
        and ``m`` is even. The padding sizes are described starting from the
        ``m/2`` to last dimension to the last dimension. That is, ``m/2``
        dimensions of input will be padded.

    value : int, optional, default=0
        Fill value for 'constant' padding
    """
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
    """
    Apply a linear transformation to the incoming data.

    .. math::
        y = x A^T + b
    """
    out = input @ weight

    if bias is not None:
        out += bias

    return out

# ---------------------- unfold ----------------------

def unfold(
    input: Tensor,
    kernel_size: _size_2_t,
    stride: _size_2_t = 1,
    padding: _size_2_t = 0,
    dilation: _size_2_t = 1
):
    """
    Extracts sliding local blocks from a batched input tensor.

    - input shape: :math:`(N, C, H, W)`
    - output shape: :math:`(N, C \\times \prod(\\text{kernel\_size}), L)`

    where:

    .. math::
        L = \prod_d \\frac{\\text{spatial\_size[d] + 2 * padding[d] - dilation[d] * (kernel\_size[d] - 1) - 1}}{\\text{stride}[d]} + 1

    where :math:`\\text{spatial\_size}` is formed by the spatial dimensions of ``input`` (H and W above),
    and :math:`d` is over all spatial dimensions.


    Parameters
    ----------
    input : Tensor
        Input tensor

    kernel_size : int or tuple
        Size of the sliding blocks.

    stride : int or tuple, optional, default=1
        Stride of the sliding blocks in the input spatial dimensions.

    padding : int or tuple, optional, default=0
        Implicit zero padding to be added on both sides of input.

    dilation : int or tuple, optional, default=1
        A parameter that controls the stride of elements within the neighborhood.
    """

    # Union[int, Tuple[int, int]] -> Tuple[int]
    kernel_size = _pair(kernel_size)
    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)

    batch_size, in_channels, h_in, w_in = input.shape
    kernel_h, kernel_w = kernel_size

    # compute the dimensions of the pooling output
    h_out = int((h_in + 2 * padding[0] - dilation[0] * (kernel_h - 1) - 1) / stride[0] + 1)
    w_out = int((w_in + 2 * padding[1] - dilation[1] * (kernel_w - 1) - 1) / stride[1] + 1)

    # padding input tensor
    padded_data = pad(input, (0, 0, 0, 0, padding[0], padding[0], padding[1], padding[1]))

    # convert input tensor and weights/kernels into a 2D matrices
    unfolded = im2col(padded_data, kernel_size, (h_out, w_out), stride, dilation)  # (batch_size, kernel_h * kernel_w * in_channels, L = h_out * w_out)

    return unfolded, h_out, w_out

# ---------------------- conv ----------------------

def conv2d(
    input: Tensor,
    weight: Tensor,
    bias: Tensor = None,
    stride: _tuple_2_t[int] = (1, 1),
    padding: _tuple_2_t[int] = (0, 0),
    dilation: _tuple_2_t[int] = (1, 1)
):
    """
    Apply a 2D convolution over an input signal composed of several input planes.

    - input shape: ``(batch_size, in_channels, h_in, w_in)``
    - output shape: ``(batch_size, out_channels, h_out, w_out)``

    where:

    .. math::
        \\text{h\_out} = \\frac{\\text{h\_in + 2 * padding[0] - dilation[0] * (kernel\_size[0] - 1) - 1}}{\\text{stride}[0]} + 1

    .. math::
        \\text{w\_out} = \\frac{\\text{w\_in + 2 * padding[1] - dilation[1] * (kernel\_size[1] - 1) - 1}}{\\text{stride}[1]} + 1

    NOTE:
        Use ``unfold`` function to perform the convolution as a single matrix multiplication. For more
        details, see [1].

    Parameters
    ----------
    input : Tensor
        Input tensor

    weight : Tensor
        Weight of the conv1d layer

    bias : Tensor, optional
        Bias of the conv2d layer

    stride : Tuple[int, int], optional, default=(1, 1)
        Stride of the convolution

    padding : Tuple[int, int], optional, default=(0, 0))
        Zero-padding added to both sides of the input

    dilation : Tuple[int, int], optional, default=(1, 1)
        Spacing between kernel elements

    References
    ----------
    1. `Why GEMM is at the heart of deep learning? Pete Warden. <https://petewarden.com/2015/04/20/why-gemm-is-at-the-heart-of-deep-learning/>`_ 2015.
    """

    batch_size, in_channels, h_in, w_in = input.shape
    out_channels, in_channels, kernel_h, kernel_w = weight.shape

    input_col, h_out, w_out = unfold(input, (kernel_h, kernel_w), stride, padding, dilation)
    input_col = input_col.permute(1, 2, 0).view(kernel_h * kernel_w * in_channels, -1)  # (kernel_h * kernel_w * in_channels, batch_size * h_out * w_out)

    weight_col = weight.view(out_channels, -1)

    out = (weight_col @ input_col).view(out_channels, h_out, w_out, batch_size).permute(3, 0, 1, 2)

    if bias is not None:
        out += bias

    return out


def conv1d(
    input: Tensor,
    weight: Tensor,
    bias: Tensor = None,
    stride: _tuple_1_t[int] = (1, ),
    padding: _tuple_1_t[int] = (0, ),
    dilation: _tuple_1_t[int] = (1, )
):
    """
    Apply a 1D convolution over an input signal composed of several input
    planes.

    - input shape: ``(batch_size, in_channels, L_in)``
    - output shape: ``(batch_size, out_channels, L_out)``

    where:

    .. math::
        \\text{L\_out} = \\frac{\\text{L\_in + 2 * padding - dilation * (kernel\_size - 1) - 1}}{\\text{stride}} + 1

    Parameters
    ----------
    input : Tensor
        Input tensor

    weight : Tensor
        Weight of the conv1d layer

    bias : Tensor, optional
        Bias of the conv1d layer

    stride : Tuple[int], optional, default: (1, )
        Stride of the convolution

    padding : Tuple[int], optional, default: (0, )
        Zero-padding added to both sides of the input

    dilation : Tuple[int], optional, default: (1, )
        Spacing between kernel elements
    """

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


# ---------------------- max pooling ----------------------

def max_pool2d(
    input: Tensor,
    kernel_size: _tuple_2_t[int],
    stride: _tuple_2_t[int],
    padding: _tuple_2_t[int] = (0, 0),
    dilation: _tuple_2_t[int] = (1, 1),
    return_indices: bool = False
):
    """
    Apply a 2D max pooling over an input signal composed of several input planes.

    - input shape: ``(batch_size, in_channels, h_in, w_in)``
    - output shape: ``(batch_size, out_channels, h_out, w_out)``

    where:

    .. math::
        \\text{h\_out} = \\frac{\\text{h\_in + 2 * padding[0] - dilation[0] * (kernel\_size[0] - 1) - 1}}{\\text{stride}[0]} + 1

    .. math::
        \\text{w\_out} = \\frac{\\text{w\_in + 2 * padding[1] - dilation[1] * (kernel\_size[1] - 1) - 1}}{\\text{stride}[1]} + 1

    NOTE:
        Use ``unfold`` function to perform the max pooling as a single matrix multiplication. For more
        details, see [1].

    NOTE:
        It should be noted that, PyTorch argues the input will be implicitly
        zero-padded when ``padding`` is non-zero in its `documentation <https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html>`_.
        However, in fact, it uses implicit **negative infinity** padding rather
        than zero-padding, see `this issue <https://github.com/pytorch/pytorch/issues/33384>`_.

        In this class, zero-padding is used.

    Parameters
    ----------
    kernel_size : Tuple[int, int]
        Size of the sliding window, must be > 0.

    stride : Tuple[int, int]
        Stride/hop of the window. Default to ``kernel_size``.

    padding : Tuple[int, int], optional, default=(0, 0)
        Zero-padding added to both sides of the input, must be >= 0 and <= ``kernel_size / 2``.

    dilation : Tuple[int, int], optional, default=(1, 1)
        Spacing between the elements in the window, must be > 0

    return_indices : bool, optional, default=False
        If ``True``, will return the max indices along with the outputs

    References
    ----------
    1. `Why GEMM is at the heart of deep learning? Pete Warden. \
        <https://petewarden.com/2015/04/20/why-gemm-is-at-the-heart-of-deep-learning/>`_ 2015.
    """

    batch_size, in_channels, h_in, w_in = input.shape
    kernel_h, kernel_w = kernel_size

    input_col, h_out, w_out = unfold(input, kernel_size, stride, padding, dilation)
    input_col = input_col.permute(1, 2, 0).view(in_channels, kernel_h * kernel_w, -1)

    out_max = input_col.max(dim=1).view(in_channels, h_out, w_out, batch_size).permute(3, 0, 1, 2)
    return out_max

def max_pool1d(
    input: Tensor,
    kernel_size: _tuple_1_t[int],
    stride: _tuple_1_t[int] = (1, ),
    padding: _tuple_1_t[int] = (0, ),
    dilation: _tuple_1_t[int] = (1, ),
    return_indices: bool = False
):
    """
    Apply a 1D max pooling over an input signal composed of several input planes.

    - input shape: ``(batch_size, in_channels, L_in)``
    - output shape: ``(batch_size, out_channels, L_out)``

    where:

    .. math::
        \\text{L\_out} = \\frac{\\text{L\_in + 2 * padding - dilation * (kernel\_size - 1) - 1}}{\\text{stride}} + 1

    NOTE:
        It should be noted that, PyTorch argues the input will be implicitly
        zero-padded when ``padding`` is non-zero in its `documentation <https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html>`_.
        However, in fact, it uses implicit **negative infinity** padding rather
        than zero-padding, see `this issue <https://github.com/pytorch/pytorch/issues/33384>`_.

        In this class, zero-padding is used.

    Parameters
    ----------
    kernel_size : Tuple[int]
        Size of the sliding window, must be > 0.

    stride : Tuple[int]
        Stride of the window, must be > 0. Default to ``kernel_size``.

    padding : Tuple[int], optional, default=0
        Zero-padding added to both sides of the input, must be >= 0 and <= ``kernel_size / 2``.

    dilation : Tuple[int], optional, default=1
        Spacing between the elements in the window, must be > 0

    return_indices : bool, optional, default=False)
        If ``True``, will return the max indices along with the outputs
    """

    # add a dimension to tensors so we can use max_pool2d
    input_2d = input.unsqueeze(dim=2)

    kernel_size_2d = (1, kernel_size)
    stride_2d = (1, stride[0])
    pad_2d = (0, padding[0])
    dilation_2d = (1, dilation[0])

    out_2d = max_pool2d(input_2d, kernel_size_2d, stride_2d, pad_2d, dilation_2d, return_indices)  # (batch_size, out_channels, 1, L_out)

    # drop the added dimension
    out = out_2d.squeeze(dim=2)
    return out

# ---------------------- dropout ----------------------

def dropout(input: Tensor, p: float = 0.5, training: bool = True) -> Tensor:
    """
    Dropout is used to randomly zeroes some of the elements of the input tensor
    with probability ``p`` using samples from a Bernoulli distribution during
    training. Furthermore, the outputs are scaled by a factor of :math:`\\frac{1}{1 - p}`
    during training. Each channel will be zeroed out independently on every forward call.

    During evaluation, the module simply computes an identity function.

    This has proven to be an effective technique for regularization and preventing
    the co-adaptation of neurons as described in the paper [1].

    Parameters
    ----------
    p : float, optional, default=0.5
        Probability of an element to be zeroed

    training : bool
        Apply dropout if is ``True``

    References
    ----------
    1. "`Improving Neural Networks by Preventing Co-adaptation of Feature Detectors. \
        <https://arxiv.org/abs/1207.0580>`_" Geoffrey E. Hinton, et al. arXiv 2012.
    """
    ret = input.data
    scaler = 1.0 / (1.0 - p)
    mask = np.random.binomial(1, 1 - p, size=input.shape)

    if training:
        ret = scaler * mask * ret

    out = Tensor(
        data = ret,
        depends_on = [input],
        requires_grad = input.requires_grad
    )

    def grad_dropout():
        if input.requires_grad:
            input.grad += scaler * mask * out.grad

    if out.requires_grad:
        out.grad_fn = grad_dropout

    return out

# ---------------------- flatten ----------------------

def flatten(input: Tensor) -> Tensor:
    """
    Flatten the input. Does not affect the batch size.

    NOTE:
        If inputs are shaped ``(batch,)`` without a feature axis, then flattening
        adds an extra channel dimension and output shape is ``(batch, 1)``.
    """
    return input.view(input.size(0), -1)
