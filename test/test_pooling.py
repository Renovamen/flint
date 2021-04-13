import os
import sys
import unittest
import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import flint

class TestPooling(unittest.TestCase):
    def test_max_pool2d(self):
        batch_size = 1
        h_in = 8
        w_in = 7
        in_channels = 1
        kernel_size = (3, 2)
        stride = (3, 2)
        pad = (0, 0)
        dilation = (2, 2)

        np.random.seed(0)
        x_init = np.random.randn(batch_size, in_channels, h_in, w_in).astype(np.float32)

        def test_flint():
            x = flint.Tensor(x_init.copy(), requires_grad=True)

            out = flint.nn.functional.max_pool2d(x, kernel_size, stride, pad, dilation)
            c = out.sum()
            c.backward()

            return out.data, x.grad

        def test_torch():
            x = torch.tensor(x_init.copy(), requires_grad=True)

            out = torch.nn.functional.max_pool2d(x, kernel_size, stride, pad, dilation)
            c = out.sum()
            c.backward()

            return out.detach().numpy(), x.grad.numpy()

        for x, y in zip(test_flint(), test_torch()):
            np.testing.assert_allclose(x, y, atol=1e-5)

if __name__ == '__main__':
    unittest.main()
