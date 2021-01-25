import sys
sys.path.append('/Users/zou/Renovamen/Developing/tinyark/')

import numpy as np
from tinyark import nn, optim, Tensor

class MLP(nn.Module):
    def __init__(self, weight1, bias1):
        super(MLP, self).__init__()

        weight1 = nn.Parameter(Tensor(weight1))
        bias1 = nn.Parameter(Tensor(bias1))

        self.l1 = nn.Linear(5, 2)
        
        self.l1.weight = weight1
        self.l1.bias = bias1

    def forward(self, x):
        out = self.l1(x)
        return out

n_epoch = 5
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
weight1 = np.random.rand(in_features, out_features)
bias1 = np.random.rand(1, out_features)

# define network
net = MLP(weight1, bias1)
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
