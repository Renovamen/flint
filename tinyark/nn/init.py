from tinyark import Tensor

def zeros_(tensor: Tensor) -> None:
    '''
    Fill the tensor with the scalar value `0`.

    args:
        tensor (Tensor): a Tensor
    '''

    tensor.zero_()

def ones_(tensor: Tensor) -> None:
    '''
    Fill the tensor with the scalar value `1`.

    args:
        tensor (Tensor): a Tensor
    '''

    tensor.one_()

def constant_(tensor: Tensor, val: float) -> None:
    '''
    Fill the tensor with the given scalar value `val`.

    args:
        tensor (Tensor): a Tensor
        val (float): the value to fill the tensor with
    '''

    tensor.fill_(val)

def uniform_(tensor: Tensor, a: float = 0., b: float = 1.) -> None:
    '''
    Fills the tensor with values drawn from the uniform distribution.

    args:
        tensor (Tensor): a Tensor
        low (float): the lower bound of the uniform distribution
        high (float): the upper bound of the uniform distribution
    '''
    tensor.uniform_(low=a, high=b)

def normal_(tensor: Tensor, mean: float = 0., std: float = 1.) -> None:
    '''
    Fills the tensor with values drawn from the normal distribution.

    args:
        tensor (Tensor): a Tensor
        mean (float): the mean of the normal distribution
        std (float): the standard deviation of the normal distribution
    '''
    tensor.normal_(mean=mean, std=std)

