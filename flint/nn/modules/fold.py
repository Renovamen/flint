from flint import Tensor
from .. import functional as F
from ..types import _size_2_t
from .module import Module


class Unfold(Module):
    """
    Extracts sliding local blocks from a batched input tensor. See :func:`flint.nn.functional.unfold`
    for more details.

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
    def __init__(
        self,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        dilation: _size_2_t = 1
    ) -> None:
        super(Unfold, self).__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.stride = stride

    def forward(self, input: Tensor) -> Tensor:
        out, _, _ = F.unfold(
            input,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation
        )
        return out
