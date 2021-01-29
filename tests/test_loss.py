import sys
sys.path.append('/Users/zou/Renovamen/Developing/tinyark/')

import unittest
import numpy as np
import tinyark
import torch

class TestLoss(unittest.TestCase):
    def test_mse(self):
        a_init = np.random.randn(3, 5).astype(np.float32)
        b_init = np.random.randn(3, 5).astype(np.float32)

        def test_tinyark():
            loss_function = tinyark.nn.MSELoss()
            
            input = tinyark.Tensor(a_init.copy(), requires_grad=True)
            target = tinyark.Tensor(b_init.copy())
            
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
        
        for x, y in zip(test_tinyark(), test_torch()):
            np.testing.assert_allclose(x, y, atol=1e-5)
    
    def test_bce(self):
        a_init = np.random.randn(3, 5).astype(np.float32)
        b_init = np.random.randn(3, 5).astype(np.float32)

        def test_tinyark():
            loss_function = tinyark.nn.BCELoss()
            sigmoid = tinyark.nn.Sigmoid()

            input = tinyark.Tensor(a_init.copy(), requires_grad=True)
            target = tinyark.Tensor(b_init.copy())

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
        
        for x, y in zip(test_tinyark(), test_torch()):
            np.testing.assert_allclose(x, y, atol=1e-5)
        
if __name__ == '__main__':
    unittest.main()
