import numpy as np

def expand_as(input: np.ndarray, target: np.ndarray, axis: int) -> np.ndarray:
	input_ndim = input.ndim
	out_orders = list(range(0, input_ndim))
	out_orders.insert(axis, input_ndim)

    target_axis_shape = target.shape[axis]
	out_shape = list(input.shape) + [target_axis_shape]
    out = input.repeat(target_axis_shape).reshape(out_shape).transpose(out_orders)
	return out

def to_categorical(target: np.ndarray, n_col: int = None) -> np.ndarray:
	'''
	Convert a class vector (integers) to binary class matrix.

	args:
        target (np.ndarray): 1-dim (N) where each value: 0 <= target[i] <= n_classes-1
        n_col (int, optional): number of colums in transformed data
	
	returns:
		one_hot (ndarray): a binary class matrix (batch_size, n_classes)
	'''
	
	n_col = n_col if n_col is not None else np.max(data) + 1
	batch_size = target.shape[0]
	one_hot = np.zeros((batch_size, n_col))
	one_hot[np.arange(batch_size), target] = 1
	return one_hot
