from flint import Tensor
from .module import Module
from .. import functional as F

class Dropout(Module):
    """
    Dropout is used to randomly zeroes some of the elements of the input tensor
    with probability ``p`` using samples from a Bernoulli distribution during
    training. Furthermore, the outputs are scaled by a factor of :math:`\\frac{1}{1 - p}`
    during training. Each channel will be zeroed out independently on every forward call.

    During evaluation, the module simply computes an identity function.

    This has proven to be an effective technique for regularization and preventing
    the co-adaptation of neurons as described in the paper [1].

    See :func:`flint.nn.functional.dropout` for more details.

    Parameters
    ----------
    p : float, optional, default=0.5
        Probability of an element to be zeroed

    References
    ----------
    1. "`Improving Neural Networks by Preventing Co-adaptation of Feature Detectors. \
        <https://arxiv.org/abs/1207.0580>`_" Geoffrey E. Hinton, et al. arXiv 2012.
    """
    def __init__(self, p: float = 0.5) -> None:
        super(Dropout, self).__init__()

        if p < 0 or p > 1:
            raise ValueError(
                "Dropout probability has to be between 0 and 1, "
                "but got {}".format(p)
            )
        self.p = p

    def forward(self, input: Tensor) -> Tensor:
        self.output = F.dropout(input, self.p)
        return self.output
