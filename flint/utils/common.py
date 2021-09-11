import numpy as np

__all__ = ['to_categorical']

def to_categorical(target: np.ndarray, n_classes: int = None) -> np.ndarray:
	"""
	Convert a class vector (integers) to binary class matrix.

	Parameters
    ----------
    target : np.ndarray
        A 1-dim (batch_size) class vector to be converted into a matrix
        (integers from 0 to n_classes - 1).

    n_classes : int, optional
        Total number of classes. If `None`, this would be inferred as the
        (largest number in target) + 1.

	Returns
    -------
	one_hot : np.ndarray
        A binary class matrix (batch_size, n_classes)
	"""
	n_classes = n_classes if n_classes is not None else np.max(target) + 1
	batch_size = target.shape[0]
	one_hot = np.zeros((batch_size, n_classes))
	one_hot[np.arange(batch_size), target] = 1
	return one_hot
