import sys
sys.path.append('/Users/zou/Renovamen/Developing/tinyark/')

import numpy as np
from tinyark import nn, optim, Tensor

class MLP(nn.Module):
    def __init__(self, weight1, bias1, weight2, bias2):
        super(MLP, self).__init__()

        self.l1 = nn.Linear(10, 5)
        self.l2 = nn.Linear(5, 2)
        self.relu = nn.ReLU()

        weight1 = nn.Parameter(Tensor(weight1))
        bias1 = nn.Parameter(Tensor(bias1))
        weight2 = nn.Parameter(Tensor(weight2))
        bias2 = nn.Parameter(Tensor(bias2))
        
        self.l1.weight = weight1
        self.l1.bias = bias1
        self.l2.weight = weight2
        self.l2.bias = bias2

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out

n_epoch = 20
lr = 0.5

np.random.seed(0)

in_features = 5
out_features = 2
batch_size = 3

# generate inputs and targets
inputs = np.random.rand(batch_size, in_features)
targets = np.random.randint(0, out_features, (batch_size, ))
x, y = Tensor(inputs), Tensor(targets)

# generate weights and bias
weight1 = np.random.rand(in_features, 5)
bias1 = np.random.rand(1, 5)
weight2 = np.random.rand(5, out_features)
bias2 = np.random.rand(1, out_features)

# define network
net = MLP(weight1, bias1, weight2, bias2)
optimer = optim.SGD(params=net.parameters(), lr=lr)
loss_function = nn.CrossEntropyLoss()

for i in range(n_epoch):
    print('\n------------------ Epoch %d ------------------' % (i + 1))
    
    optimer.zero_grad()

    pred = net(x)
    # print('Prediction: ', pred.data)

    loss = loss_function(pred, y)
    print('Loss: ', loss.data)

    loss.backward()
    optimer.step()
