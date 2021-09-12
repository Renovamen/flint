from flint import Tensor
from .. import functional as F
from .module import Module

__all__ = [
    'ReLU',
    'LeakyReLU',
    'Sigmoid',
    'Tanh',
    'GELU'
]


class ReLU(Module):
    """
    ReLU (Rectified Linear Unit) activation function. See :func:`flint.nn.functional.relu` for more
    details.
    """
    def __init__(self) -> None:
        super(ReLU, self).__init__()

    def forward(self, input: Tensor) -> Tensor:
        self.data = F.relu(input)
        return self.data


class LeakyReLU(Module):
    """
    Leaky ReLU activation function. See :func:`flint.nn.functional.leaky_relu` for more details.

    .. math::
        \\text{LeakyReLU}(x) = \max(0, x) + \\text{negative\_slope} * \min(0, x)

    Parameters
    ----------
    negative_slope : float, optional, default=1e-2
        Controls the angle of the negative slope.
    """
    def __init__(self, negative_slope: float = 1e-2) -> None:
        super(ReLU, self).__init__()
        self.negative_slope = negative_slope

    def forward(self, input: Tensor) -> Tensor:
        self.data = F.leaky_relu(input, self.negative_slope)
        return self.data


class Sigmoid(Module):
    """
    Sigmoid activation function. See :func:`flint.nn.functional.sigmoid` for more details.

    .. math::
        \\text{sigmoid}(x) = \\frac{1}{1 + \exp(-x)}
    """
    def __init__(self) -> None:
        super(Sigmoid, self).__init__()

    def forward(self, input: Tensor) -> Tensor:
        self.data = F.sigmoid(input)
        return self.data


class Tanh(Module):
    """
    Tanh (Hyperbolic Tangent) activation function. See :func:`flint.nn.functional.tanh` for more details.

    .. math::
        \\text{tanh}(x) = \\frac{\sinh(x)}{\cosh(x)} = \\frac{\exp(x) - \exp(-x)}{\exp(x) + \exp(-x)}
    """
    def __init__(self) -> None:
        super(Tanh, self).__init__()

    def forward(self, input: Tensor) -> Tensor:
        self.data = F.tanh(input)
        return self.data


class GELU(Module):
    """
    Gaussian Error Linear Units (GELU) function. See :func:`flint.nn.functional.gelu` for more details.

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
    def __init__(self) -> None:
        super(GELU, self).__init__()

    def forward(self, input: Tensor) -> Tensor:
        self.data = F.gelu(input)
        return self.data
