from flint import Tensor
from .module import Module
from .. import functional as F

class Flatten(Module):
    """
    Flatten the input. Does not affect the batch size.

    NOTE:
        If inputs are shaped ``(batch,)`` without a feature axis, then flattening
        adds an extra channel dimension and output shape is ``(batch, 1)``.
    """
    def __init__(self) -> None:
        super(Flatten, self).__init__()

    def forward(self, input: Tensor) -> Tensor:
        self.output = F.flatten(input)
        return self.output
