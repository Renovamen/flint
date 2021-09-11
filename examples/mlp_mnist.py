"""
This is an example for showing how to train a MLP using Flint on the MNIST dataset.
"""

import os
import sys

# A temporary solution for relative imports in case flint is not installed.
# If flint is installed, the following line is not needed.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from flint import nn, optim

from utils import get_data
from runners import train, test


class MLP(nn.Module):
    def __init__(self, in_features: int, n_classes: int):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, in_features // 2),
            nn.ReLU(),
            nn.Linear(in_features // 2, in_features // 4),
            nn.ReLU(),
            nn.Linear(in_features // 4, n_classes)
        )

    def forward(self, x):
        out = self.model(x)
        return out


if __name__ == '__main__':
    # ---- hyper parameters ----
    n_epochs = 10
    batch_size = 128
    lr = 0.001
    in_features = 28 * 28
    n_classes = 10
    print_freq = 100
    # --------------------------

    # initialize your network
    net = MLP(in_features, n_classes)
    # optimizer
    optimer = optim.Adam(params=net.parameters(), lr=lr)
    # loss function
    loss_function = nn.CrossEntropyLoss()
    # dataset
    train_loader, test_loader = get_data(batch_size)

    # start training!
    train(n_epochs, train_loader, net, optimer, loss_function, print_freq)

    # test the model
    test(test_loader, net)
