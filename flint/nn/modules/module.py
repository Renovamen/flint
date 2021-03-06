from collections import OrderedDict
from typing import Optional, Union, Iterator, Tuple, Set
from flint import Tensor
from .. import Parameter

class Module(object):
    """
    Base class for all modules.

    Args:
        name (str): name of the module
    """

    training: bool

    def __init__(self) -> None:
        self.training = True
        self._parameters = OrderedDict()
        self._modules = OrderedDict()

    def register_parameter(self, name: str, param: Optional[Parameter]) -> None:
        """
        Add a parameter to the module.

        Args:
            name (str): name of the parameter
            param (Parameter): parameter to be added to the module
        """
        if param is None:
            self._parameters[name] = None
        else:
            self._parameters[name] = param

    def add_module(self, name: str, module: Optional['Module']) -> None:
        """
        Add a child module to the current module.

        Args:
            name (str): name of the child module
            module (Module): child module to be added to the module
        """
        if module is None:
            self._modules[name] = None
        else:
            self._modules[name] = module

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        """
        Returns an iterator over module parameters, only yielding the parameter itself.

        Args:
            recurse (bool): If ``True``, yields parameters of this module and all
                submodules. If ``False``, yields only parameters that are direct
                members of this module.

        Yields:
            Parameter: module parameter
        """
        for name, param in self.named_parameters(recurse = recurse):
            yield param

    def named_parameters(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, Parameter]]:
        """
        Returns an iterator over module parameters, yielding both the name of the
        parameter as well as the parameter itself.

        Adapted from: https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/module.py

        Args:
            prefix (str): prefix to prepend to all parameter names.
            recurse (bool):
                True: yield parameters of this module and all submodules
                False: yield only parameters that are direct members of this module

        Yields:
            (string, Parameter): Tuple containing the name and parameter
        """

        memo = set()
        modules = self.named_modules(prefix = prefix) if recurse else [(prefix, self)]
        for module_prefix, module in modules:
            params = module._parameters.items()
            for k, v in params:
                if v is None or v in memo:
                    continue
                memo.add(v)
                name = module_prefix + ('.' if module_prefix else '') + k
                yield name, v

    def children(self) -> Iterator['Module']:
        """
        Returns an iterator over immediate children modules.

        Yields:
            module (Module): A child module
        """
        for name, module in self.named_children():
            yield module

    def named_children(self) -> Iterator[Tuple[str, 'Module']]:
        """
        Returns an iterator over immediate children modules, yielding both the
        name of the module as well as the module itself.

        Yields:
            (string, Module): Tuple containing a name and child module
        """
        memo = set()
        for name, module in self._modules.items():
            if module is not None and module not in memo:
                memo.add(module)
                yield name, module

    def modules(self) -> Iterator['Module']:
        """
        Returns an iterator over all modules in the network, only yielding the module itself.

        yields:
            Module: a module in the network

        NOTE:
            Duplicate modules are returned only once.
        """
        for name, module in self.named_modules():
            yield module

    def named_modules(self, memo: Optional[Set['Module']] = None, prefix: str = '') -> Iterator[Tuple[str, 'Module']]:
        """
        Returns an iterator over all modules in the network, yielding
        both the name of the module as well as the module itself.
        Borrowed from: https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/module.py

        Args:
            memo (Set): a set for recording visited modules
            prefix (str): prefix to prepend to all parameter names

        yields:
            (string, Module): Tuple of name and module

        NOTE:
            Duplicate modules are returned only once.
        """
        if memo is None:
            memo = set()
        if self not in memo:
            memo.add(self)
            yield prefix, self
            for name, module in self._modules.items():
                if module is None:
                    continue
                submodule_prefix = prefix + ('.' if prefix else '') + name
                for m in module.named_modules(memo, submodule_prefix):
                    yield m

    def train(self, mode: bool = True) -> 'Module':
        """
        Sets the module in training mode.

        This has effect only on the following modules:

        - :class:`flint.nn.Dropout`

        See their documentations for details of their behaviors in training /
        evaluation mode.

        Parameters
        ----------
        mode : bool, optional, default=True
            Whether to set training mode (``True``) or evaluation mode (``False``)

        Returns
        -------
        module : Module
        """
        self.training = mode
        for module in self.children():
            module.train(mode)
        return self

    def eval(self) -> 'Module':
        """
        Sets the module in evaluation mode.

        This has effect only on the following modules:

        - :class:`flint.nn.Dropout`

        See their documentations for details of their behaviors in training /
        evaluation mode.

        Returns
        -------
        module : Module
        """
        return self.train(False)

    def __call__(self, *input, **kwargs) -> Tensor:
        out = self.forward(*input, **kwargs)
        return out

    def __setattr__(self, name: str, value):
        # add a parameter to the module
        if isinstance(value, Parameter):
            self.register_parameter(name, value)
        # add a child module to the module
        elif isinstance(value, Module):
            self.add_module(name, value)
        object.__setattr__(self, name, value)

    def __delattr__(self, name):
        # delete a parameter from the module
        if name in self._parameters:
            del self._parameters[name]
        # delete a child module from the module
        elif name in self.modules:
            del self._modules[name]
