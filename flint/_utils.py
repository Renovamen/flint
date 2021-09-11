import numpy as np

def unbroadcast_add(input: np.ndarray, other: np.ndarray) -> np.ndarray:
    unmatched_axis = [i for i, s in enumerate(other.shape) if s != input.shape[i]]
    for axis in unmatched_axis:
        other = other.sum(axis=axis, keepdims=True)
    return input + other
