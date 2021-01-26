import numpy as np

def to_categorical(target: np.ndarray, n_col: int = None) -> np.ndarray:
	'''
	Convert a class vector (integers) to binary class matrix.

	args:
        target (np.ndarray): 1-dim (N) where each value: 0 <= target[i] <= n_classes-1
        n_col (int, optional): number of colums in transformed data
	
	returns:
		one_hot (ndarray): a binary class matrix (batch_size, n_classes)
	'''
	
	n_col = n_col if n_col is not None else np.max(target) + 1
	batch_size = target.shape[0]
	one_hot = np.zeros((batch_size, n_col))
	one_hot[np.arange(batch_size), target] = 1
	return one_hot

def broadcast_add(input: np.ndarray, other: np.ndarray) -> np.ndarray:
	unmatched_axis = [i for i, s in enumerate(other.shape) if s != input.shape[i]]
	if unmatched_axis != []:
		return input + np.sum(other, axis=unmatched_axis[0], keepdims=True)
	else:
		return input + other
