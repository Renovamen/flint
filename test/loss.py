import numpy as np

x = np.array([
    [0.2, 2, 0.3],
    [0.1, 4, 0.5]
], dtype = np.float32)

target = np.array([0, 2], dtype = np.int)

batch_size = x.shape[0]

# print(np.arange(batch_size))

a = x[np.arange(batch_size), target.astype(np.int)]

print(a)