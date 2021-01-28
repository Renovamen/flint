from tinyark import Tensor
from .. import functional as F

class ReLU:
    '''
    ReLU (Rectified Linear Unit) activation function
    See tinyark.nn.functional.relu() for more details.
    '''

    def __init__(self) -> None:
        super(ReLU, self).__init__()

    def __call__(self, input: Tensor) -> Tensor:
        self.data = F.relu(input)
        return self.data

class Sigmoid:
    '''
    Sigmoid activation function
    See tinyark.nn.functional.sigmoid() for more details.
    '''

    def __init__(self) -> None:
        super(Sigmoid, self).__init__()

    def __call__(self, input: Tensor) -> Tensor:
        self.data = F.sigmoid(input)
        return self.data
    
class Tanh:
    '''
    Tanh (Hyperbolic Tangent) activation function
    See tinyark.nn.functional.tanh() for more details.
    '''

    def __init__(self) -> None:
        super(Tanh, self).__init__()

    def __call__(self, input: Tensor) -> Tensor:
        self.data = F.tanh(input)
        return self.data
