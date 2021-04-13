from typing import Optional

from flint import Tensor
from .. import functional as F
from ..types import _size_1_t, _size_2_t, _tuple_any_t
from ._utils import _single, _pair
from .module import Module

class _MaxPoolNd(Module):
    """
    A base class for all types of max pooling layers.

    Args:
        kernel_size (tuple): Size of the window to take a max over
        stride (tuple): Stride/hop of the window
        padding (tuple): Zero-padding added to both sides of the input
        dilation (tuple): Spacing between the elements in the window
        return_indices (bool, optional, default=False): If ``True``, will return
            the max indices along with the outputs
    """

    def __init__(
        self,
        kernel_size: _tuple_any_t[int],
        stride: _tuple_any_t[int],
        padding: _tuple_any_t[int],
        dilation: _tuple_any_t[int],
        return_indices: bool = False
    ) -> None:
        super(_MaxPoolNd, self).__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices


class MaxPool1d(_MaxPoolNd):
    """
    Apply a 1D max pooling over an input signal composed of several input planes.
    See :func:`flint.nn.functional.maxpool1d` for more details.

    NOTE:
        It should be noted that, PyTorch argues the input will be implicitly
        zero-padded when ``padding`` is non-zero in its `documentation <https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html>`_.
        However, in fact, it uses implicit **negative infinity** padding rather
        than zero-padding, see `this issue <https://github.com/pytorch/pytorch/issues/33384>`_.

        In this class, zero-padding is used.

    Args:
        kernel_size (_size_1_t): Size of the sliding window, must be > 0.
        stride (_size_1_t): Stride of the window, must be > 0. Default to ``kernel_size``.
        padding (_size_1_t, optional, default=0): Zero-padding added to both
            sides of the input, must be >= 0 and <= ``kernel_size / 2``.
        dilation (_size_1_t, optional, default=1): Spacing between the elements
            in the window, must be > 0
        return_indices (bool, optional, default=False): If ``True``, will return
            the max indices along with the outputs

    Shapes:
        - input: (batch_size, in_channels, L_in)
        - output: (batch_size, out_channels, L_out)

        where:

        .. math::
            \\text{L\_out} = \\frac{\\text{L\_in + 2 * padding - dilation * (kernel\_size - 1) - 1}}{\\text{stride}} + 1
    """

    def __init__(
        self,
        kernel_size: _size_1_t,
        stride: _size_1_t = 1,
        padding: _size_1_t = 0,
        dilation: _size_1_t = 1,
        return_indices: bool = False
    ):
        # Union[int, Tuple[int]] -> Tuple[int]
        kernel_size_ = _single(kernel_size)
        stride_ = _single(stride)
        padding_ = _single(padding)
        dilation_ = _single(dilation)

        super(MaxPool1d, self).__init__(
            kernel_size = kernel_size_,
            stride = stride_,
            padding = padding_,
            dilation = dilation_,
            return_indices = return_indices
        )

    def forward(self, input: Tensor) -> Tensor:
        return F.max_pool1d(
            input,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.return_indices
        )


class MaxPool2d(_MaxPoolNd):
    """
    Apply a 2D max pooling over an input signal composed of several input planes.
    See :func:`flint.nn.functional.maxpool2d` for more details.

    NOTE:
        It should be noted that, PyTorch argues the input will be implicitly
        zero-padded when ``padding`` is non-zero in its `documentation <https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html>`_.
        However, in fact, it uses implicit **negative infinity** padding rather
        than zero-padding, see `this issue <https://github.com/pytorch/pytorch/issues/33384>`_.

        In this class, zero-padding is used.

    Args:
        kernel_size (_size_2_t): Size of the sliding window, must be > 0.
        stride (_size_2_t): Stride of the window, must be > 0. Default to ``kernel_size``.
        padding (_size_2_t, optional, default=0): Zero-padding added to both
            sides of the input, must be >= 0 and <= ``kernel_size / 2``.
        dilation (_size_2_t, optional, default=1): Spacing between the elements
            in the window, must be > 0
        return_indices (bool, optional, default=False): If ``True``, will return
            the max indices along with the outputs

    Shape:
        - input: (batch_size, in_channels, h_in, w_in)
        - output: (batch_size, out_channels, h_out, w_out)

        where:

        .. math::
            \\text{h\_out} = \\frac{\\text{h\_in + 2 * padding[0] - dilation[0] * (kernel\_size[0] - 1) - 1}}{\\text{stride}[0]} + 1

        .. math::
            \\text{w\_out} = \\frac{\\text{w\_in + 2 * padding[1] - dilation[1] * (kernel\_size[1] - 1) - 1}}{\\text{stride}[1]} + 1
    """

    def __init__(
        self,
        kernel_size: _size_2_t,
        stride: Optional[_size_2_t] = None,
        padding: _size_2_t = 0,
        dilation: _size_2_t = 1,
        return_indices: bool = False
    ):
        # Union[int, Tuple[int, int]] -> Tuple[int, int]
        kernel_size_ = _pair(kernel_size)
        if stride:
            stride_ = _pair(stride)
        else:
            stride_ = kernel_size_
        padding_ = _pair(padding)
        dilation_ = _pair(dilation)

        super(MaxPool2d, self).__init__(
            kernel_size = kernel_size_,
            stride = stride_,
            padding = padding_,
            dilation = dilation_,
            return_indices = return_indices
        )

    def forward(self, input: Tensor) -> Tensor:
        return F.max_pool2d(
            input,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.return_indices
        )
