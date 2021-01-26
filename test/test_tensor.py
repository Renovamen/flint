import sys
sys.path.append('/Users/zou/Renovamen/Developing/tinyark/')

import unittest
import numpy as np
import tinyark
import torch

class TestTensor(unittest.TestCase):
    def test_add(self):
        def test_tinyark():
            a = tinyark.Tensor(2.0, requires_grad=True)
            b = tinyark.Tensor(3.0, requires_grad=True)
            c = a + b
            c.backward()
            return c.data, a.grad, b.grad

        def test_torch():
            a = torch.tensor(2.0, requires_grad=True)
            b = torch.tensor(3.0, requires_grad=True)
            c = a + b
            c.backward()
            return c.detach().numpy(), a.grad.numpy(), b.grad.numpy()
        
        for x, y in zip(test_tinyark(), test_torch()):
            np.testing.assert_allclose(x, y, atol=1e-5)
        
    def test_sub(self):
        def test_tinyark():
            a = tinyark.Tensor(2.0, requires_grad=True)
            b = tinyark.Tensor(3.0, requires_grad=True)
            c = a - b
            c.backward()
            return c.data, a.grad, b.grad

        def test_torch():
            a = torch.tensor(2.0, requires_grad=True)
            b = torch.tensor(3.0, requires_grad=True)
            c = a - b
            c.backward()
            return c.detach().numpy(), a.grad.numpy(), b.grad.numpy()
        
        for x, y in zip(test_tinyark(), test_torch()):
            np.testing.assert_allclose(x, y, atol=1e-5)

    def test_neg(self):
        def test_tinyark():
            a = tinyark.Tensor(2.0, requires_grad=True)
            b = -a
            b.backward()
            return b.data, a.grad

        def test_torch():
            a = torch.tensor(2.0, requires_grad=True)
            b = -a
            b.backward()
            return b.detach().numpy(), a.grad.numpy()
        
        for x, y in zip(test_tinyark(), test_torch()):
            np.testing.assert_allclose(x, y, atol=1e-5)
        
    def test_mul(self):
        def test_tinyark():
            a = tinyark.Tensor(2.0, requires_grad=True)
            b = tinyark.Tensor(3.0, requires_grad=True)
            c = a * b
            c.backward()
            return c.data, a.grad, b.grad

        def test_torch():
            a = torch.tensor(2.0, requires_grad=True)
            b = torch.tensor(3.0, requires_grad=True)
            c = a * b
            c.backward()
            return c.detach().numpy(), a.grad.numpy(), b.grad.numpy()
        
        for x, y in zip(test_tinyark(), test_torch()):
            np.testing.assert_allclose(x, y, atol=1e-5)
    
    def test_div(self):
        def test_tinyark():
            a = tinyark.Tensor(2.0, requires_grad=True)
            b = tinyark.Tensor(3.0, requires_grad=True)
            c = a / b
            c.backward()
            return c.data, a.grad, b.grad

        def test_torch():
            a = torch.tensor(2.0, requires_grad=True)
            b = torch.tensor(3.0, requires_grad=True)
            c = a / b
            c.backward()
            return c.detach().numpy(), a.grad.numpy(), b.grad.numpy()
        
        for x, y in zip(test_tinyark(), test_torch()):
            np.testing.assert_allclose(x, y, atol=1e-5)
    
    def test_pow(self):
        def test_tinyark():
            a = tinyark.Tensor(2.0, requires_grad=True)
            b = a ** 2
            b.backward()
            return b.data, a.grad

        def test_torch():
            a = torch.tensor(2.0, requires_grad=True)
            b = a ** 2
            b.backward()
            return b.detach().numpy(), a.grad.numpy()
        
        for x, y in zip(test_tinyark(), test_torch()):
            np.testing.assert_allclose(x, y, atol=1e-5)
    
    def test_log(self):
        def test_tinyark():
            a = tinyark.Tensor(2.0, requires_grad=True)
            b = a.log()
            b.backward()
            return b.data, a.grad

        def test_torch():
            a = torch.tensor(2.0, requires_grad=True)
            b = a.log()
            b.backward()
            return b.detach().numpy(), a.grad.numpy()
        
        for x, y in zip(test_tinyark(), test_torch()):
            np.testing.assert_allclose(x, y, atol=1e-5)
    
    def test_exp(self):
        def test_tinyark():
            a = tinyark.Tensor(2.0, requires_grad=True)
            b = a.exp()
            b.backward()
            return b.data, a.grad

        def test_torch():
            a = torch.tensor(2.0, requires_grad=True)
            b = a.exp()
            b.backward()
            return b.detach().numpy(), a.grad.numpy()
        
        for x, y in zip(test_tinyark(), test_torch()):
            np.testing.assert_allclose(x, y, atol=1e-5)

    def test_sum(self):
        def test_tinyark():
            a = tinyark.Tensor([1., 2., 3., 4.], requires_grad=True)
            b = a ** 2
            c = b.sum()
            c.backward()
            return c.data, a.grad

        def test_torch():
            a = torch.tensor([1., 2., 3., 4.], requires_grad=True)
            b = a ** 2
            c = b.sum()
            c.backward()
            return c.detach().numpy(), a.grad.numpy()
        
        for x, y in zip(test_tinyark(), test_torch()):
            np.testing.assert_allclose(x, y, atol=1e-5)
    
    def test_max(self):
        def test_tinyark():
            a = tinyark.Tensor([1., 2., 4., 4.], requires_grad=True)
            b = a ** 2
            c = b.max()
            c.backward()
            return c.data, a.grad

        def test_torch():
            a = torch.tensor([1., 2., 4., 4.], requires_grad=True)
            b = a ** 2
            c = b.max()
            c.backward()
            return c.detach().numpy(), a.grad.numpy()
        
        for x, y in zip(test_tinyark(), test_torch()):
            np.testing.assert_allclose(x, y, atol=1e-5)
    
    def test_dot(self):
        a_init = np.random.randn(1, 3).astype(np.float32)
        b_init = np.random.randn(3, 3).astype(np.float32)

        def test_tinyark():
            a = tinyark.Tensor(a_init, requires_grad=True)
            b = tinyark.Tensor(b_init, requires_grad=True)
            c = a @ b
            d = c.sum()
            d.backward()
            return c.data, d.data, a.grad, b.grad

        def test_torch():
            a = torch.tensor(a_init, requires_grad=True)
            b = torch.tensor(b_init, requires_grad=True)
            c = a @ b
            d = c.sum()
            d.backward()
            return c.detach().numpy(), d.detach().numpy(), a.grad.numpy(), b.grad.numpy()
        
        for x, y in zip(test_tinyark(), test_torch()):
            np.testing.assert_allclose(x, y, atol=1e-5)
    
    def test_softmax(self):
        a_init = np.random.randn(5).astype(np.float32)

        def test_tinyark():
            a = tinyark.Tensor(a_init, requires_grad=True)
            b = a.softmax()
            c = b.sum()
            c.backward()
            return c.data, b.data, a.grad

        def test_torch():
            a = torch.tensor(a_init, requires_grad=True)
            b = torch.nn.functional.softmax(a, dim=0)
            c = b.sum()
            c.backward()
            return c.detach().numpy(), b.detach().numpy(), a.grad.numpy()
        
        for x, y in zip(test_tinyark(), test_torch()):
            np.testing.assert_allclose(x, y, atol=1e-5)
        
    def test_log_softmax(self):
        a_init = np.random.randn(5).astype(np.float32)

        def test_tinyark():
            a = tinyark.Tensor(a_init, requires_grad=True)
            b = a.log_softmax()
            c = b.sum()
            c.backward()
            return c.data, b.data, a.grad

        def test_torch():
            a = torch.tensor(a_init, requires_grad=True)
            b = torch.nn.functional.log_softmax(a, dim=0)
            c = b.sum()
            c.backward()
            return c.detach().numpy(), b.detach().numpy(), a.grad.numpy()
        
        for x, y in zip(test_tinyark(), test_torch()):
            np.testing.assert_allclose(x, y, atol=1e-5)
            
        
if __name__ == '__main__':
    unittest.main()
