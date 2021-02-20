import os
import sys
import unittest
import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import flint


class TestLoss(unittest.TestCase):
    def test_mse(self):
        a_init = np.random.randn(3, 5).astype(np.float32)
        b_init = np.random.randn(3, 5).astype(np.float32)

        def test_flint():
            loss_function = flint.nn.MSELoss()

            input = flint.Tensor(a_init.copy(), requires_grad=True)
            target = flint.Tensor(b_init.copy())

            loss = loss_function(input, target)
            loss.backward()

            return loss.data, input.grad

        def test_torch():
            loss_function = torch.nn.MSELoss()

            input = torch.tensor(a_init.copy(), requires_grad=True)
            target = torch.tensor(b_init.copy())

            loss = loss_function(input, target)
            loss.backward()

            return loss.detach().numpy(), input.grad.numpy()

        for x, y in zip(test_flint(), test_torch()):
            np.testing.assert_allclose(x, y, atol=1e-5)

    def test_bce(self):
        a_init = np.random.randn(3, 5).astype(np.float32)
        b_init = np.random.randn(3, 5).astype(np.float32)

        def test_flint():
            loss_function = flint.nn.BCELoss()
            sigmoid = flint.nn.Sigmoid()

            input = flint.Tensor(a_init.copy(), requires_grad=True)
            target = flint.Tensor(b_init.copy())

            loss = loss_function(sigmoid(input), target)
            loss.backward()
            return loss.data, input.grad

        def test_torch():
            loss_function = torch.nn.BCELoss()
            sigmoid = torch.nn.Sigmoid()

            input = torch.tensor(a_init.copy(), requires_grad=True)
            target = torch.tensor(b_init.copy())

            loss = loss_function(sigmoid(input), target)
            loss.backward()

            return loss.detach().numpy(), input.grad.numpy()

        for x, y in zip(test_flint(), test_torch()):
            np.testing.assert_allclose(x, y, atol=1e-5)

if __name__ == '__main__':
    unittest.main()
