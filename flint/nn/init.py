'''
Some of the code is borrowed from: https://github.com/pytorch/pytorch/blob/master/torch/nn/init.py
'''
import math
import numpy as np
from typing import Optional, Union

from flint import Tensor

def calculate_gain(nonlinearity: str, param: Optional[Union[int, float]] = None):
    """
    Return the recommended gain value for the given nonlinearity function.

    The values are as follows:

    ================= ====================================================
    nonlinearity      gain
    ================= ====================================================
    Linear / Identity :math:`1`
    Conv{1,2,3}D      :math:`1`
    Sigmoid           :math:`1`
    Tanh              :math:`\\frac{5}{3}`
    ReLU              :math:`\sqrt{2}`
    Leaky Relu        :math:`\sqrt{\\frac{2}{1 + \\text{negative\_slope}^2}}`
    SELU              :math:`\\frac{3}{4}`
    ================= ====================================================

    Parameters
    ----------
    nonlinearity : str
        Name of the non-linear function

    param : Union[int, float], optional
        Optional parameter for the non-linear function
    """

    linear_fns = ['linear', 'conv1d', 'conv2d', 'conv3d']
    if nonlinearity in linear_fns or nonlinearity == 'sigmoid':
        return 1
    elif nonlinearity == 'tanh':
        return 5.0 / 3
    elif nonlinearity == 'relu':
        return math.sqrt(2.0)
    elif nonlinearity == 'leaky_relu':
        if param is None:
            negative_slope = 0.01
        elif not isinstance(param, bool) and isinstance(param, int) or isinstance(param, float):
            # True/False are instances of int, hence check above
            negative_slope = param
        else:
            raise ValueError("negative_slope {} not a valid number".format(param))
        return math.sqrt(2.0 / (1 + negative_slope ** 2))
    elif nonlinearity == 'selu':
        return 3.0 / 4  # Value found empirically (https://github.com/pytorch/pytorch/pull/50664)
    else:
        raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))


def zeros_(tensor: Tensor) -> None:
    """
    Fill the tensor with the scalar value ``0``.

    Args:
        tensor (Tensor): A Tensor
    """
    tensor.zero_()

def ones_(tensor: Tensor) -> None:
    """
    Fill the tensor with the scalar value ``1``.

    Args:
        tensor (Tensor): A Tensor
    """
    tensor.one_()

def constant_(tensor: Tensor, val: float) -> None:
    """
    Fill the tensor with the given scalar value ``val``.

    Args:
        tensor (Tensor): A Tensor
        val (float): The value to fill the tensor with
    """
    tensor.fill_(val)

def uniform_(tensor: Tensor, a: float = 0., b: float = 1.) -> None:
    """
    Fills the tensor with values drawn from the uniform distribution.

    Args:
        tensor (Tensor): A Tensor
        low (float): The lower bound of the uniform distribution
        high (float): The upper bound of the uniform distribution
    """
    tensor.uniform_(low=a, high=b)

def normal_(tensor: Tensor, mean: float = 0., std: float = 1.) -> None:
    """
    Fills the tensor with values drawn from the normal distribution.

    Args:
        tensor (Tensor): A Tensor
        mean (float): The mean of the normal distribution
        std (float): The standard deviation of the normal distribution
    """
    tensor.normal_(mean=mean, std=std)


def _calculate_fan_in_and_fan_out(tensor: Tensor):
    """
    Compute number of input and output nodes for a tensor.

    Parameters
    ----------
    tensor : Tensor
        A Tensor

    Returns
    -------
    fan_in : int
        Number of input nodes

    fan_out : int
        Number of output nodes
    """
    dimensions = tensor.ndim
    if dimensions < 2:
        raise ValueError('Fan in and fan out can not be computed for tensor with fewer than 2 dimensions')

    num_input_fmaps = tensor.shape[1]
    num_output_fmaps = tensor.shape[0]
    receptive_field_size = 1
    if dimensions > 2:
        receptive_field_size = np.prod(tensor.shape[2:])
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out

def xavier_uniform_(tensor: Tensor, gain: float = 1.) -> None:
    """
    Implementation of Xavier initialization proposed in [1]. Also known
    as Glorot initialization, using a uniform distribution.

    The resulting tensor will have values sampled from :math:`U(-a, a)`,
    where ``a = gain * sqrt(6 / (fan_in + fan_out))``.

    Parameters
    ----------
    tensor : Tensor
        A Tensor

    gain : float, optional, default=1.
        An optional scaling factor

    References
    ----------
    1. "`Understanding the Difficulty of Training Deep Feedforward Neural Networks. <http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf>`_" Xavier Glorot and Yoshua Bengio. AISTATS 2010.
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    a = math.sqrt(3.0) * std  # calculate uniform bounds from standard deviation

    tensor.uniform_(low=-a, high=a)

def xavier_normal_(tensor: Tensor, gain: float = 1.) -> None:
    """
    Implementation of Xavier initialization proposed in [1]. Also known
    as Glorot initialization, using a normal distribution.

    The resulting tensor will have values sampled from :math:`N(0, \\text{std}^2)`,
    where ``std = gain * sqrt(2 / (fan_in + fan_out))``

    Parameters
    ----------
    tensor : Tensor
        A Tensor

    gain : float, optional, default=1.
        An optional scaling factor

    References
    ----------
    1. "`Understanding the Difficulty of Training Deep Feedforward Neural Networks. <http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf>`_" Xavier Glorot and Yoshua Bengio. AISTATS 2010.
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))

    tensor.normal_(mean=0, std=std)


def _calculate_correct_fan(tensor: Tensor, mode: str):
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out']
    if mode not in valid_modes:
        raise ValueError("Mode {} not supported, please use one of {}".format(mode, valid_modes))

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == 'fan_in' else fan_out

def kaiming_uniform_(
    tensor: Tensor,
    a: float = 0.,
    mode: str = 'fan_in',
    nonlinearity: str = 'leaky_relu'
) -> None:
    """
    Implementation of Kaiming initialization proposed in [1]. Also known
    as He initialization, using a uniform distribution.

    The resulting tensor will have values sampled from :math:`U(-\\text{bound}, \\text{bound})`,
    where ``bound = gain * sqrt*(3 / fan_mode)``.

    Parameters
    ----------
    tensor : Tensor
        A Tensor

    a : float, optional, default=0.
        The negative slope of the rectifier used after this layer (only used
        with 'leaky_relu')

    mode : str, optional, default='fan_in'
        Either ``'fan_in'`` or ``'fan_out'``. ``'fan_in'`` for preserving the
        magnitude of the variance of the weights in the forward pass. ``'fan_out'``
        for preserving the magnitudes in the backwards pass.

    nonlinearity : str, optional, default='leaky_relu'
        Name of the non-linear function, recommended to use only with 'relu'
        or 'leaky_relu'

    References
    ----------
    1. "`Delving Deep into Rectifiers: Surpassing Human-level Performance on ImageNet Classification. <https://arxiv.org/pdf/1502.01852.pdf>`_" Kaiming He, et al. ICCV 2015.
    """
    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std  # calculate uniform bounds from standard deviation

    tensor.uniform_(low=-bound, high=bound)

def kaiming_normal_(
    tensor: Tensor,
    a: float = 0.,
    mode: str = 'fan_in',
    nonlinearity: str = 'leaky_relu'
) -> None:
    """
    Implementation of Kaiming initialization proposed in [1]. Also known
    as He initialization, using a normal distribution.

    The resulting tensor will have values sampled from :math:`N(0, \\text{std}^2)`,
    where ``std = gain / sqrt(fan_mode)``.

    Parameters
    ----------
    tensor : Tensor
        A Tensor

    a : float, optional, default=0.
        The negative slope of the rectifier used after this layer (only used
        with 'leaky_relu')

    mode : str, optional, default='fan_in'
        Either ``'fan_in'`` or ``'fan_out'``. ``'fan_in'`` for preserving the
        magnitude of the variance of the weights in the forward pass. ``'fan_out'``
        for preserving the magnitudes in the backwards pass.

    nonlinearity : str, optional, default='leaky_relu'
        Name of the non-linear function, recommended to use only with 'relu'
        or 'leaky_relu'

    References
    ----------
    1. "`Delving Deep into Rectifiers: Surpassing Human-level Performance on ImageNet Classification. <https://arxiv.org/pdf/1502.01852.pdf>`_" Kaiming He, et al. ICCV 2015.
    """
    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)

    tensor.normal_(mean=0, std=std)


def lecun_uniform_(tensor: Tensor) -> None:
    """
    Implementation of LeCun initialization, using a uniform distribution.

    The resulting tensor will have values sampled from :math:`U(-\\text{bound}, \\text{bound})`,
    where ``bound = sqrt(3 / fan_in)``.

    Args:
        tensor (Tensor): A Tensor

    References
    ----------
    1. "`Efficient Backprop. <http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf>`_" Yann LeCun, et al. 1998.
    """
    fan_in, _ = _calculate_fan_in_and_fan_out(tensor)
    bound = math.sqrt(3.0 / fan_in)  # calculate uniform bounds from standard deviation

    tensor.uniform_(low=-bound, high=bound)

def lecun_normal_(tensor: Tensor) -> None:
    """
    Implementation of LeCun initialization, using a normal distribution.

    The resulting tensor will have values sampled from :math:`N(0, \\text{std}^2)`,
    where ``std = sqrt(1 / fan_in)``.

    Args:
        tensor (Tensor): A Tensor

    References
    ----------
    1. "`Efficient Backprop. <http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf>`_" Yann LeCun, et al. 1998.
    """
    fan_in, _ = _calculate_fan_in_and_fan_out(tensor)
    std = math.sqrt(1.0 / fan_in)

    tensor.normal_(mean=0, std=std)
