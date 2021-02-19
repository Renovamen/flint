# adopted from: https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/container.py

from collections import OrderedDict
from typing import overload, Iterator

from flint import Tensor
from .module import Module

class Sequential(Module):
    '''
    A sequential container. Modules will be added to it in the order they
    are passed in the constructor. Alternatively, an ordered dict of modules
    can also be passed in.
    '''

    @overload
    def __init__(self, *args: Module) -> None:
        ...

    @overload
    def __init__(self, arg: 'OrderedDict[str, Module]') -> None:
        ...

    def __init__(self, *args):
        super(Sequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self) -> Iterator[Module]:
        return iter(self._modules.values())

    def forward(self, input: Tensor) -> Tensor:
        for module in self:
            input = module(input)
        return input
