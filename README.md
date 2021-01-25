# TinyArk

A toy deep learning framework built with Numpy from scratch with a PyTorch-like API.

&nbsp;

## Installation

```bash
git clone https://github.com/Renovamen/tinyark.git
cd tinyark
python setup.py install
```

or

```bash
pip install git+https://github.com/Renovamen/tinyark.git --upgrade
```

&nbsp;

## Example

Build your net first:

```python
from tinyark import nn

class Net(nn.Module):
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
```

Then you can train it:

```python
from tinyark import nn, optim, Tensor
import numpy as np

n_epoch = 20
lr = 0.5

np.random.seed(0)

# specify some parameters
in_features = 10
out_features = 2
batch_size = 5

# randomly generate inputs and targets
inputs = np.random.rand(batch_size, in_features)
targets = np.random.randint(0, out_features, (batch_size, ))
x, y = Tensor(inputs), Tensor(targets)

# initialize your network
net = MLP(weight1, bias1, weight2, bias2)
# choose an optimizer and a loss function
optimer = optim.SGD(params=net.parameters(), lr=lr)
loss_function = nn.CrossEntropyLoss()

# then we can train it
for i in range(n_epoch):

    optimer.zero_grad()

    pred = net(x)

    loss = loss_function(pred, y)
    loss.backward()
    
    optimer.step()

    print(
        'Epoch: [{0}][{1}]\t'
        'Loss {loss:.4f}\t'.format(i + 1, n_epoch, loss = loss.data)
    )
```

&nbsp;

## Features

### Core

Support autograd on the following operations:

- [x] Add
- [x] Substract
- [x] Negative
- [x] Muliply
- [x] Divide
- [x] Matmul
- [x] Power
- [x] Natural Logarithm
- [x] Exponential
- [x] Sum
- [x] Softmax
- [x] Log Softmax

### Layers

- [x] Linear
- [ ] Conv
- [ ] Flatten
- [ ] MaxPooling
- [ ] Dropout
- [ ] BatchNormalization
- [ ] RNN

### Optimizers

- [x] SGD
- [ ] Momentum
- [ ] AdaGrad
- [ ] RMSprop
- [ ] AdaDelta
- [ ] Adam

### Loss Functions

- [x] Cross Entropy
- [x] Negative Log Likelihood
- [ ] Mean Squared Error

### Activation Functions

- [x] ReLU
- [x] Sigmoid
- [x] Tanh

### Initializers

- [ ] Uniform / Normal
- [ ] Xavier Uniform / Normal
- [ ] Kaiming Uniform / Normal

&nbsp;

## License

[MIT](LICENSE)

&nbsp;

## Acknowledgements

This project is inspired by [karpathy/micrograd](https://github.com/karpathy/micrograd).
