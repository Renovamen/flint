import os
import sys
sys.path.append(os.getcwd())

import unittest
from typing import Tuple
import numpy as np
import torch
import flint

in_features = 3
out_features = 3
batch_size = 1

x_init = np.random.randn(batch_size, in_features).astype(np.float32)
w_init = np.random.randn(in_features, out_features).astype(np.float32)
b_init = np.random.randn(1, out_features).astype(np.float32)

class FlintNet(flint.nn.Module):
    def __init__(self):
        super(FlintNet, self).__init__()
        self.x = flint.Tensor(x_init.copy())
        self.w = flint.nn.Parameter(flint.Tensor(w_init.copy()))
        self.b = flint.nn.Parameter(flint.Tensor(b_init.copy()))
        self.relu = flint.nn.ReLU()

    def forward(self) -> flint.Tensor:
        out = self.x @ self.w
        out = self.relu(out).log_softmax()
        out = (out + self.b).sum()
        return out

class TorchNet(torch.nn.Module):
    def __init__(self):
        super(TorchNet, self).__init__()
        self.x = torch.tensor(x_init.copy())
        self.w = torch.nn.Parameter(torch.tensor(w_init.copy()))
        self.b = torch.nn.Parameter(torch.tensor(b_init.copy()))
        self.relu = torch.nn.ReLU()

    def forward(self) -> torch.tensor:
        out = self.x.matmul(self.w)
        out = self.relu(out)
        out = torch.nn.functional.log_softmax(out, dim=1)
        out = out.add(self.b).sum()
        return out


def step_flint(optim: flint.optim.Optimizer, kwargs = {}) -> Tuple[np.ndarray, np.ndarray]:
    net = FlintNet()
    optim = optim([net.w, net.b], **kwargs)
    out = net.forward()
    out.backward()
    optim.step()
    return net.w.data

def step_pytorch(optim: torch.optim.Optimizer, kwargs = {}) -> Tuple[np.ndarray, np.ndarray]:
    net = TorchNet()
    optim = optim([net.x, net.w], **kwargs)
    out = net.forward()
    out.backward()
    optim.step()
    return net.w.detach().numpy()


class TestOptim(unittest.TestCase):
    def test_sgd(self):
        x = step_flint(flint.optim.SGD, kwargs={'lr': 0.001, 'weight_decay': 0.1})
        y = step_pytorch(torch.optim.SGD, kwargs={'lr': 0.001, 'weight_decay': 0.1})
        np.testing.assert_allclose(x, y, atol=1e-5)

    def test_momentum(self):
        # heavy ball / polyak's momentum
        x = step_flint(flint.optim.SGD, kwargs={'lr': 0.001, 'momentum': 0.01})
        y = step_pytorch(torch.optim.SGD, kwargs={'lr': 0.001, 'momentum': 0.01})
        np.testing.assert_allclose(x, y, atol=1e-5)

        # nesterov's momentum
        x = step_flint(flint.optim.SGD, kwargs={'lr': 0.001, 'momentum': 0.01, 'nesterov': True})
        y = step_pytorch(torch.optim.SGD, kwargs={'lr': 0.001, 'momentum': 0.01, 'nesterov': True})
        np.testing.assert_allclose(x, y, atol=1e-5)

    def test_adagrad(self):
        x = step_flint(flint.optim.Adagrad, kwargs={'lr': 0.001})
        y = step_pytorch(torch.optim.Adagrad, kwargs={'lr': 0.001})
        np.testing.assert_allclose(x, y, atol=1e-5)

    def test_rmsprop(self):
        x = step_flint(flint.optim.RMSprop, kwargs={'lr': 0.001, 'alpha': 0.95})
        y = step_pytorch(torch.optim.RMSprop, kwargs={'lr': 0.001, 'alpha': 0.95})
        np.testing.assert_allclose(x, y, atol=1e-5)

    def test_adadelta(self):
        x = step_flint(flint.optim.Adadelta, kwargs={'lr': 0.01, 'rho': 0.97})
        y = step_pytorch(torch.optim.Adadelta, kwargs={'lr': 0.01, 'rho': 0.97})
        np.testing.assert_allclose(x, y, atol=1e-5)

    def test_adam(self):
        x = step_flint(flint.optim.Adam)
        y = step_pytorch(torch.optim.Adam)
        np.testing.assert_allclose(x, y, atol=1e-5)


if __name__ == '__main__':
    unittest.main()
