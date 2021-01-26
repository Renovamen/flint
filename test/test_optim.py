import sys
sys.path.append('/Users/zou/Renovamen/Developing/tinyark/')

import unittest
from typing import Tuple
import numpy as np
import torch
import tinyark

in_features = 3
out_features = 3
batch_size = 1

x_init = np.random.randn(batch_size, in_features).astype(np.float32)
w_init = np.random.randn(in_features, out_features).astype(np.float32)
b_init = np.random.randn(1, out_features).astype(np.float32)

class TinyarkNet(tinyark.nn.Module):
    def __init__(self):
        super(TinyarkNet, self).__init__()
        self.x = tinyark.Tensor(x_init.copy())
        self.w = tinyark.nn.Parameter(tinyark.Tensor(w_init.copy()))
        self.b = tinyark.nn.Parameter(tinyark.Tensor(b_init.copy()))
        self.relu = tinyark.nn.ReLU()

    def forward(self) -> tinyark.Tensor:
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


def step_tinyark(optim: tinyark.optim.Optimizer, kwargs = {}) -> Tuple[np.ndarray, np.ndarray]:
    net = TinyarkNet()
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
        for x, y in zip(
            step_tinyark(tinyark.optim.SGD, kwargs={'lr': 0.001}),
            step_pytorch(torch.optim.SGD, kwargs={'lr': 0.001})
        ):
            np.testing.assert_allclose(x, y, atol=1e-5)


if __name__ == '__main__':
    unittest.main()
