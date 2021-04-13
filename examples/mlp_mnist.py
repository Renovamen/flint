'''
This is an example for showing how to a MLP on the MNIST dataset.
'''

import os
import sys
from tqdm import tqdm
import numpy as np
import torchvision

# A temporary solution for relative imports in case flint is not installed.
# If flint is installed, the following line is not needed.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import flint
from flint import nn, optim, Tensor


class MLP(nn.Module):
    def __init__(self, in_features: int, n_classes: int):
        super(MLP, self).__init__()
        self.in_features = in_features
        self.model = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.ReLU(),
            nn.Linear(in_features // 2, in_features // 4),
            nn.ReLU(),
            nn.Linear(in_features // 4, n_classes)
        )

    def forward(self, x: Tensor):
        x = x.view(-1, self.in_features)
        out = self.model(x)
        return out

def collate_fn(batch):
    elem = batch[0]
    if isinstance(elem, int):
        return Tensor(np.array(batch))
    elif isinstance(elem, tuple):
        return tuple(collate_fn(samples) for samples in zip(*batch))
    else:
        return Tensor(np.stack([samples.numpy() for samples in batch]))

def get_data(batch_size: int):
    trans = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (1.0,))
    ])

    train_data = torchvision.datasets.MNIST(
        root = "./data/",
        transform = trans,
        train = True,
        download = True
    )
    test_data = torchvision.datasets.MNIST(
        root = "./data/",
        transform = trans,
        train = False
    )

    train_loader = flint.utils.data.DataLoader(
        dataset = train_data,
        batch_size = batch_size,
        collate_fn = collate_fn
    )
    test_loader = flint.utils.data.DataLoader(
        dataset = test_data,
        batch_size = batch_size,
        collate_fn = collate_fn
    )

    return train_loader, test_loader

def train(n_epochs, train_loader, net, optimer, loss_function, print_freq):
    for epoch in range(n_epochs):
        for i, batch in enumerate(train_loader):
            images, labels = batch

            # clear gradients
            optimer.zero_grad()

            # forward prop.
            scores = net(images)

            # compute loss and do backward prop.
            loss = loss_function(scores, labels)
            loss.backward()

            # update weights
            optimer.step()

            # compute accuracy
            preds = scores.argmax(dim=1)
            correct_preds = flint.eq(preds, labels).sum().data
            accuracy = correct_preds / labels.shape[0]

            # print training status
            if i % print_freq == 0:
                print(
                    'Epoch: [{0}][{1}/{2}]\t'
                    'Loss {loss:.4f}\t'
                    'Accuracy {acc:.3f}'.format(
                        epoch + 1, i + 1, len(train_loader),
                        loss = loss.data,
                        acc = accuracy
                    )
                )

def test(test_loader, net):
    for i, batch in enumerate(tqdm(test_loader, desc = 'Testing')):
        images, labels = batch
        scores = net(images)

        # compute accuracy
        preds = scores.argmax(dim=1)
        correct_preds = flint.eq(preds, labels).sum().data
        accuracy = correct_preds / labels.shape[0]

    print('\n * TEST ACCURACY - %.1f percent\n' % (accuracy * 100))

if __name__ == '__main__':
    # define parameters here
    n_epochs = 10
    batch_size = 128
    lr = 0.001
    in_features = 28 * 28
    n_classes = 10
    print_freq = 100

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
