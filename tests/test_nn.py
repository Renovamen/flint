import os
import sys
sys.path.append(os.getcwd())

import unittest
import numpy as np
import flint
import torch

class TestNN(unittest.TestCase):
    def test_linear(self):
        batch_size = 2
        in_features = 10
        out_features = 3

        np.random.seed(0)
        x_init = np.random.randn(batch_size, in_features).astype(np.float32)
        w_init = np.random.randn(in_features, out_features).astype(np.float32)
        b_init = np.random.randn(1, out_features).astype(np.float32)

        def test_flint():
            x = flint.Tensor(x_init.copy(), requires_grad=True)
            w = flint.Tensor(w_init.copy(), requires_grad=True)
            b = flint.Tensor(b_init.copy(), requires_grad=True)

            out = flint.nn.functional.linear(x, w, b)
            c = out.sum()
            c.backward()

            return out.data, w.grad

        def test_torch():
            x = torch.tensor(x_init.copy(), requires_grad=True)
            w = torch.tensor(w_init.transpose(1, 0).copy(), requires_grad=True)
            b = torch.tensor(b_init.squeeze().copy(), requires_grad=True)

            out = torch.nn.functional.linear(x, w, b)
            c = out.sum()
            c.backward()

            return out.detach().numpy(), w.grad.permute(1, 0).numpy()

        for x, y in zip(test_flint(), test_torch()):
            np.testing.assert_allclose(x, y, atol=1e-5)

    def test_conv2d(self):
        batch_size = 2
        h_in = 8
        w_in = 7
        in_channels = 1
        out_channels = 2
        kernel_size = (3, 2)
        stride = (3, 2)
        pad = (2, 2)
        dilation = (2, 2)

        np.random.seed(0)
        x_init = np.random.randn(batch_size, in_channels, h_in, w_in).astype(np.float32)
        w_init = np.random.randn(out_channels, in_channels, kernel_size[0], kernel_size[1]).astype(np.float32)
        b_init = np.random.randn(1, out_channels, 1, 1).astype(np.float32)

        def test_flint():
            x = flint.Tensor(x_init.copy(), requires_grad=True)
            w = flint.Tensor(w_init.copy(), requires_grad=True)
            b = flint.Tensor(b_init.copy(), requires_grad=True)

            out = flint.nn.functional.conv2d(x, w, b, stride, pad, dilation)
            c = out.sum()
            c.backward()

            return out.data, w.grad

        def test_torch():
            x = torch.tensor(x_init.copy(), requires_grad=True)
            w = torch.tensor(w_init.copy(), requires_grad=True)
            b = torch.tensor(np.squeeze(b_init.copy()), requires_grad=True)

            out = torch.nn.functional.conv2d(x, w, b, stride, pad, dilation)
            c = out.sum()
            c.backward()

            return out.detach().numpy(), w.grad.numpy()

        for x, y in zip(test_flint(), test_torch()):
            np.testing.assert_allclose(x, y, atol=1e-5)

    def test_conv1d(self):
        batch_size = 2
        l_in = 9
        in_channels = 1
        out_channels = 2
        kernel_size = 3
        stride = (2, )
        pad = (1, )
        dilation = (2, )

        np.random.seed(0)
        x_init = np.random.randn(batch_size, in_channels, l_in).astype(np.float32)
        w_init = np.random.randn(out_channels, in_channels, kernel_size).astype(np.float32)
        b_init = np.random.randn(1, out_channels, 1).astype(np.float32)

        def test_flint():
            x = flint.Tensor(x_init.copy(), requires_grad=True)
            w = flint.Tensor(w_init.copy(), requires_grad=True)
            b = flint.Tensor(b_init.copy(), requires_grad=True)

            out = flint.nn.functional.conv1d(x, w, b, stride, pad, dilation)
            c = out.sum()
            c.backward()

            return out.data, w.grad

        def test_torch():
            x = torch.tensor(x_init.copy(), requires_grad=True)
            w = torch.tensor(w_init.copy(), requires_grad=True)
            b = torch.tensor(np.squeeze(b_init.copy()), requires_grad=True)

            out = torch.nn.functional.conv1d(x, w, b, stride, pad, dilation)
            c = out.sum()
            c.backward()

            return out.detach().numpy(), w.grad.numpy()

        for x, y in zip(test_flint(), test_torch()):
            np.testing.assert_allclose(x, y, atol=1e-5)


if __name__ == '__main__':
    unittest.main()
