from ..tensor import Tensor

class Parameter(Tensor):
    def __init__(self, t: Tensor):
        super(Parameter, self).__init__(t.name, t.shape, t.requires_grad, t.dtype)
        self.data = t.data
