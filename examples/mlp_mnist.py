'''
This is an example for showing how to a MLP on the MNIST dataset.
'''

import sys
sys.path.append('/Users/zou/Renovamen/Developing/tinyark/')

import numpy as np
import torchvision
import tinyark
from tinyark import nn, optim, Tensor

class MLP(nn.Module):
    def __init__(self, in_features: int, n_classes: int):
        super(MLP, self).__init__()
        self.in_features = in_features
        self.l1 = nn.Linear(in_features, in_features // 2)
        self.l2 = nn.Linear(in_features // 2, in_features // 4)
        self.l3 = nn.Linear(in_features // 4, n_classes)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor):
        batch_size = x.shape[0]
        x = x.view(-1, self.in_features)
        out = self.relu(self.l1(x))
        out = self.relu(self.l2(out))
        out = self.l3(out)
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

    train_loader = tinyark.utils.data.DataLoader(
        dataset = train_data,
        batch_size = batch_size,
        collate_fn = collate_fn
    )
    test_loader = tinyark.utils.data.DataLoader(
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

            # find accuracy
            preds = scores.argmax(axis = 1)
            correct_preds = tinyark.eq(preds, labels).sum().data
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