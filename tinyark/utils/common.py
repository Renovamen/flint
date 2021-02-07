import numpy as np

def to_categorical(target: np.ndarray, n_classes: int = None) -> np.ndarray:
	'''
	Convert a class vector (integers) to binary class matrix.

	args:
        target (np.ndarray): A 1-dim (batch_size) class vector to be converted
			into a matrix (integers from 0 to n_classes - 1).
        n_classes (int, optional): Total number of classes. If `None`, this
			would be inferred as the (largest number in target) + 1.

	returns:
		one_hot (ndarray): a binary class matrix (batch_size, n_classes)
	'''

	n_classes = n_classes if n_classes is not None else np.max(target) + 1
	batch_size = target.shape[0]
	one_hot = np.zeros((batch_size, n_classes))
	one_hot[np.arange(batch_size), target] = 1
	return one_hot

def broadcast_add(input: np.ndarray, other: np.ndarray) -> np.ndarray:
	unmatched_axis = [i for i, s in enumerate(other.shape) if s != input.shape[i]]
	if unmatched_axis != []:
		return input + np.sum(other, axis=unmatched_axis[0], keepdims=True)
	else:
		return input + other
