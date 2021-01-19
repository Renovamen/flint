from tinyark import Tensor

class Parameter(Tensor):
    def __init__(self, t: Tensor):
        super(Parameter, self).__init__(t.name, t.data, requires_grad=true)
