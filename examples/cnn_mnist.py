"""
This is an example for showing how to train a CNN using Flint on the MNIST dataset.
"""

import os
import sys

# A temporary solution for relative imports in case flint is not installed.
# If flint is installed, the following line is not needed.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from flint import nn, optim

from utils import get_data
from runners import train, test


class CNN(nn.Module):
    def __init__(self, n_channels: int, n_classes: int):
        super(CNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # (batch_size, 32, 14, 14)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # (batch_size, 64, 7, 7)
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(64 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        out = self.model(x)
        return out


if __name__ == '__main__':
    # ---- hyper parameters ----
    n_epochs = 10
    batch_size = 128
    lr = 0.001
    n_channels = 1
    n_classes = 10
    print_freq = 10
    # --------------------------

    # initialize your network
    net = CNN(n_channels, n_classes)

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
