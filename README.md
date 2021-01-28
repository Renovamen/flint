# TinyArk

A toy deep learning framework built with Numpy from scratch with a [PyTorch](https://github.com/pytorch/pytorch)-like API.

I'm trying to make it as clean as possible.

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

## Documentation

Coming soon 咕咕咕...


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

# training parameters
n_epoch = 20
lr = 0.5
batch_size = 5

# model parameters
in_features = 10
out_features = 2

# randomly generate inputs and targets
inputs = np.random.rand(batch_size, in_features)
targets = np.random.randint(0, n_classes, (batch_size, ))
x, y = Tensor(inputs), Tensor(targets)

# initialize your network
net = Net(in_features, n_classes)

# choose an optimizer and a loss function
optimer = optim.SGD(params=net.parameters(), lr=lr)
loss_function = nn.CrossEntropyLoss()

# then we can train it
for i in range(n_epoch):
    # clear gradients
    optimer.zero_grad()
    
    # forward prop.
    pred = net(x)

    # compute loss and do backward prop.
    loss = loss_function(pred, y)
    loss.backward()
    
    optimer.step()

    print(
        'Epoch: [{0}][{1}]\t'
        'Loss {loss:.4f}\t'.format(i + 1, n_epoch, loss = loss.data)
    )
```

&nbsp;

## Features / To-Do List

### Core

Support autograding on the following operations:

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
- [x] Max
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
- [ ] Sequential

### Optimizers

- [x] SGD
- [x] Momentum
- [x] Adagrad
- [x] RMSprop
- [x] Adadelta
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

- [x] Fill with zeros / ones / other given constants
- [x] Uniform / Normal
- [x] Xavier (Glorot) uniform / normal ([Understanding the Difficulty of Training Deep Feedforward Neural Networks.](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) *Xavier Glorot and Yoshua Bengio.* AISTATS 2010.)
- [x] Kaiming (He) uniform / normal ([Delving Deep into Rectifiers: Surpassing Human-level Performance on ImageNet Classification.](https://arxiv.org/pdf/1502.01852.pdf) *Kaiming He, et al.* ICCV 2015.)
- [x] LeCun uniform / normal ([Efficient Backprop.](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf) *Yann LeCun, et al.* 1998.)


### Others

- [ ] Maybe incorporate the [Keras](https://github.com/keras-team/keras)-like API to make the training process simpler?


&nbsp;

## License

[MIT](LICENSE)

&nbsp;

## Acknowledgements

This project is inspired by [karpathy/micrograd](https://github.com/karpathy/micrograd).
