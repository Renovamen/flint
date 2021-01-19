import numpy as np
from tinyark import Tensor
from tinyark.nn import Parameter
from .module import Module

class Linear(Module):
    '''
    Full connected layer: y = xA^T + b
    
    args:
        in_features (int): size of each input sample
        out_features (int): size of each output sample
        bias (bool): enable bias or not
    '''
    def __init__(self, in_features: int, out_features: int, use_bias: bool = True) -> None:
        super(Linear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = Parameter(Tensor.zeros(out_features, in_features))
        if bias:
            self.bias = Parameter(Tensor.zeros(out_features, 1))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x: Tensor) -> Tensor:
        np.dot(x, self.weight, out = self.output)
        
        if self.bias is not None:
            np.add(self.output, self.bias, out = self.output)

        return self.output
