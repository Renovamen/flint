import os
import sys
sys.path.append(os.getcwd())

import unittest
import numpy as np
import tinyark
import torch

a_init = np.random.randn(5).astype(np.float32)

class TestActivators(unittest.TestCase):
    def test_relu(self):
        def test_tinyark():
            a = tinyark.Tensor(a_init.copy(), requires_grad=True)
            b = tinyark.nn.functional.relu(a)
            c = b.sum()
            c.backward()
            return c.data, b.data, a.grad

        def test_torch():
            a = torch.tensor(a_init.copy(), requires_grad=True)
            b = torch.relu(a)
            c = b.sum()
            c.backward()
            return c.detach().numpy(), b.detach().numpy(), a.grad.numpy()
        
        for x, y in zip(test_tinyark(), test_torch()):
            np.testing.assert_allclose(x, y, atol=1e-5)
    
    def test_sigmoid(self):
        def test_tinyark():
            a = tinyark.Tensor(a_init.copy(), requires_grad=True)
            b = tinyark.nn.functional.sigmoid(a)
            c = b.sum()
            c.backward()
            return c.data, b.data, a.grad

        def test_torch():
            a = torch.tensor(a_init.copy(), requires_grad=True)
            b = torch.sigmoid(a)
            c = b.sum()
            c.backward()
            return c.detach().numpy(), b.detach().numpy(), a.grad.numpy()
        
        for x, y in zip(test_tinyark(), test_torch()):
            np.testing.assert_allclose(x, y, atol=1e-5)
    
    def test_tanh(self):
        def test_tinyark():
            a = tinyark.Tensor(a_init.copy(), requires_grad=True)
            b = tinyark.nn.functional.tanh(a)
            c = b.sum()
            c.backward()
            return c.data, b.data, a.grad

        def test_torch():
            a = torch.tensor(a_init.copy(), requires_grad=True)
            b = torch.tanh(a)
            c = b.sum()
            c.backward()
            return c.detach().numpy(), b.detach().numpy(), a.grad.numpy()
        
        for x, y in zip(test_tinyark(), test_torch()):
            np.testing.assert_allclose(x, y, atol=1e-5)
        
        
if __name__ == '__main__':
    unittest.main()
