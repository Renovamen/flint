import sys
sys.path.append('/Users/zou/Renovamen/Developing/tinyark/')

import numpy as np
from tinyark import nn, optim, Tensor

n_epoch = 20
lr = 0.5
batch_size = 5
in_features = 10
n_classes = 2

class MLP(nn.Module):
    def __init__(self, in_features, n_classes):
        super(MLP, self).__init__()
        self.l1 = nn.Linear(in_features, 5)
        self.l2 = nn.Linear(5, n_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out

def get_data():
    np.random.seed(0)
    inputs = np.random.rand(batch_size, in_features)
    targets = np.random.randint(0, n_classes, (batch_size, ))
    return Tensor(inputs), Tensor(targets)

if __name__ == '__main__':
    net = MLP(in_features, n_classes)
    optimer = optim.SGD(params=net.parameters(), lr=lr)
    loss_function = nn.CrossEntropyLoss()
    
    x, y = get_data()

    for i in range(n_epoch):
        optimer.zero_grad()

        pred = net(x)
        # print('Prediction: ', pred.data)

        loss = loss_function(pred, y)
        loss.backward()
        
        optimer.step()

        print(
            'Epoch: [{0}][{1}]\t'
            'Loss {loss:.4f}\t'.format(i + 1, n_epoch, loss = loss.data)
        )
