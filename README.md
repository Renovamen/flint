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
import tinyark
from tinyark import nn, optim, Tensor
import numpy as np

# training parameters
n_epoch = 20
lr = 0.001
batch_size = 5

# model parameters
in_features = 10
out_features = 2

# generate some fake data
inputs = np.random.rand(batch_size, in_features)
targets = np.random.randint(0, n_classes, (batch_size, ))
x, y = Tensor(inputs), Tensor(targets)

# initialize your network
net = Net(in_features, n_classes)

# choose an optimizer and a loss function
optimer = optim.Adam(params=net.parameters(), lr=lr)
loss_function = nn.CrossEntropyLoss()

# then we can train it
for i in range(n_epoch):
    # clear gradients
    optimer.zero_grad()
    
    # forward prop.
    scores = net(x)

    # compute loss and do backward prop.
    loss = loss_function(scores, y)
    loss.backward()
    
    # update weights
    optimer.step()

    # compute accuracy
    preds = scores.argmax(axis = 1)
    correct_preds = tinyark.eq(preds, labels).sum().data
    accuracy = correct_preds / labels.shape[0]

    # print training status
    print(
        'Epoch: [{0}][{1}/{2}]\t'
        'Loss {loss:.4f}\t'
        'Accuracy {acc:.3f}'.format(
            epoch + 1, i + 1, len(train_loader),
            loss = loss.data,
            acc = accuracy
        )
    )
```

Check the [`examples`](examples) folder for more detailed examples.


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
- [x] View
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
- [x] Adam

### Loss Functions

- [x] Cross Entropy
- [x] Negative Log Likelihood
- [x] Mean Squared Error
- [x] Binary Cross Entropy

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

- [x] Dataloaders
- [ ] Support GPU
- [ ] Maybe incorporate the [Keras](https://github.com/keras-team/keras)-like API to make the training code simpler?


&nbsp;

## License

[MIT](LICENSE)

&nbsp;

## Acknowledgements

TinyArk is inspired by the following projects:

- [PyTorch](https://github.com/pytorch/pytorch)
- [karpathy/micrograd](https://github.com/karpathy/micrograd)
- [teddykoker/tinyloader](https://github.com/teddykoker/tinyloader)
