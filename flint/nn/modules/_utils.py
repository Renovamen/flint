from itertools import repeat

__all__ = [
    '_single',
    '_pair',
    '_triple',
    '_quadruple'
]

def _ntuple(n):
    def parse(x):
        if isinstance(x, int):
            return tuple(repeat(x, n))
        elif isinstance(x, tuple):
            assert len(x) is n
            return x
        else:
            raise ValueError
    return parse

_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)
