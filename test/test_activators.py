import os
import sys
import unittest
import numpy as np
import torch

# A temporary solution for relative imports in case flint is not installed.
# If flint is installed, the following line is not needed.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import flint


a_init = np.random.randn(5).astype(np.float32)

class TestActivators(unittest.TestCase):
    def test_relu(self):
        def test_flint():
            a = flint.Tensor(a_init.copy(), requires_grad=True)
            b = flint.nn.functional.relu(a)
            c = b.sum()
            c.backward()
            return c.data, b.data, a.grad

        def test_torch():
            a = torch.tensor(a_init.copy(), requires_grad=True)
            b = torch.relu(a)
            c = b.sum()
            c.backward()
            return c.detach().numpy(), b.detach().numpy(), a.grad.numpy()

        for x, y in zip(test_flint(), test_torch()):
            np.testing.assert_allclose(x, y, atol=1e-5)

    def test_leaky_relu(self):
        def test_flint():
            a = flint.Tensor(a_init.copy(), requires_grad=True)
            b = flint.nn.functional.leaky_relu(a)
            c = b.sum()
            c.backward()
            return c.data, b.data, a.grad

        def test_torch():
            a = torch.tensor(a_init.copy(), requires_grad=True)
            b = torch.nn.functional. leaky_relu(a)
            c = b.sum()
            c.backward()
            return c.detach().numpy(), b.detach().numpy(), a.grad.numpy()

        for x, y in zip(test_flint(), test_torch()):
            np.testing.assert_allclose(x, y, atol=1e-5)

    def test_sigmoid(self):
        def test_flint():
            a = flint.Tensor(a_init.copy(), requires_grad=True)
            b = flint.nn.functional.sigmoid(a)
            c = b.sum()
            c.backward()
            return c.data, b.data, a.grad

        def test_torch():
            a = torch.tensor(a_init.copy(), requires_grad=True)
            b = torch.sigmoid(a)
            c = b.sum()
            c.backward()
            return c.detach().numpy(), b.detach().numpy(), a.grad.numpy()

        for x, y in zip(test_flint(), test_torch()):
            np.testing.assert_allclose(x, y, atol=1e-5)

    def test_tanh(self):
        def test_flint():
            a = flint.Tensor(a_init.copy(), requires_grad=True)
            b = flint.nn.functional.tanh(a)
            c = b.sum()
            c.backward()
            return c.data, b.data, a.grad

        def test_torch():
            a = torch.tensor(a_init.copy(), requires_grad=True)
            b = torch.tanh(a)
            c = b.sum()
            c.backward()
            return c.detach().numpy(), b.detach().numpy(), a.grad.numpy()

        for x, y in zip(test_flint(), test_torch()):
            np.testing.assert_allclose(x, y, atol=1e-5)

    def test_gelu(self):
        def test_flint():
            a = flint.Tensor(a_init.copy(), requires_grad=True)
            b = flint.nn.functional.gelu(a)
            c = b.sum()
            c.backward()
            return c.data, b.data, a.grad

        def test_torch():
            a = torch.tensor(a_init.copy(), requires_grad=True)
            b = torch.nn.functional.gelu(a)
            c = b.sum()
            c.backward()
            return c.detach().numpy(), b.detach().numpy(), a.grad.numpy()

        for x, y in zip(test_flint(), test_torch()):
            np.testing.assert_allclose(x, y, atol=1e-3)


if __name__ == '__main__':
    unittest.main()
