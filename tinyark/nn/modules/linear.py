import numpy as np
import math

from tinyark import Tensor
from tinyark.nn import Parameter, init
from .module import Module

class Linear(Module):
    '''
    Full connected layer: y = xA^T + b
    
    args:
        in_features (int): size of each input sample
        out_features (int): size of each output sample
        bias (bool): enable bias or not
    '''
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(Linear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = Parameter(Tensor.zeros(in_features, out_features))
        if bias:
            self.bias = Parameter(Tensor.zeros(1, out_features))
        else:
            self.register_parameter('bias', None)
        
        self.init_parameters()
    
    def init_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        self.output = input @ self.weight

        if self.bias is not None:
            self.output += self.bias
            
        return self.output
