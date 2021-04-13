"""
Create some useful type aliases
"""

from typing import TypeVar, Union, Tuple

T = TypeVar('T')

_tuple_any_t = Tuple[T, ...]
_tuple_1_t = Tuple[T]
_tuple_2_t = Tuple[T, T]

_scalar_or_tuple_any_t = Union[T, _tuple_any_t[T]]
_scalar_or_tuple_1_t = Union[T, _tuple_1_t[T]]
_scalar_or_tuple_2_t = Union[T, _tuple_2_t[T]]

# For arguments which represent size parameters (eg, kernel size, padding)
_size_any_t = _scalar_or_tuple_any_t[int]
_size_1_t = _scalar_or_tuple_1_t[int]
_size_2_t = _scalar_or_tuple_2_t[int]
