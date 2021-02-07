import math
from typing import Union, Tuple

from tinyark import Tensor
from .. import Parameter, init
from .module import Module
from .utils import _single, _pair, _triple
from .. import functional as F

_Single = Union[int, Tuple[int]]
_Pair = Union[int, Tuple[int, int]]


class _ConvNd(Module):
    '''
    A base class for all types of conv layers.

    args:
        in_channels (int): number of channels in the input image
        out_channels (int): number of channels produced by the convolution
        kernel_size (tuple): size of the convolving kernel
        stride (tuple): stride of the convolution
        padding (tuple): zero-padding added to both sides of the input
        dilation (tuple): spacing between kernel elements
        bias (bool): enable bias or not
    '''

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple,
        stride: Tuple,
        padding: Tuple,
        dilation: Tuple,
        bias: bool
    ) -> None:
        super(_ConvNd, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.weight = Parameter(Tensor.zeros(out_channels, in_channels, *kernel_size))

        if bias:
            self.bias = Parameter(Tensor.zeros(1, out_channels, *[1 for k in kernel_size]))
        else:
            self.register_parameter('bias', None)

        self.init_parameters()

    def init_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

class Conv2d(_ConvNd):
    '''
    Apply a 2D convolution over an input signal composed of several input
    planes. See nn.functional.conv2d for more details.

    args:
        in_channels (int): number of channels in the input image
        out_channels (int): number of channels produced by the convolution
        kernel_size (int or tuple): size of the convolving kernel
        stride (int or tuple[int, int], optional):
            stride of the convolution (default: 1)
        padding (int or tuple[int, int], optional):
            zero-padding added to both sides of the input (default: 0)
        dilation (int or tuple[int, int], optional):
            spacing between kernel elements (default: 1)
        bias (bool, optional): enable bias or not (default: True)

    shape:
        input: (batch_size, in_channels, h_in, w_in)
        output: (batch_size, out_channels, h_out, w_out)

        where:
            h_out = (h_in + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1
            w_out = (w_in + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1
    '''

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _Pair,
        stride: _Pair = 1,
        padding: _Pair = 0,
        dilation: _Pair = 1,
        bias: bool = True
    ):
        # Union[int, Tuple[int, int]] -> Tuple[int]
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = _pair(padding)
        dilation_ = _pair(dilation)

        super(Conv2d, self).__init__(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = kernel_size_,
            stride = stride_,
            padding = padding_,
            dilation = dilation_,
            bias = bias
        )

    def forward(self, input: Tensor) -> Tensor:
        return F.conv2d(
            input,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation
        )
