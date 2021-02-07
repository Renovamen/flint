from itertools import repeat

def _single():
    if isinstance(x, int):
        return x
    elif isinstance(x, tuple):
        assert len(x) is 1
        return x[0]
    else:
        raise ValueError

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

_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)
