![homemade-pytorch](docs/_static/img/meme.png)

# Flint

A toy deep learning framework implemented in Numpy from scratch with a [PyTorch](https://github.com/pytorch/pytorch)-like API. I'm trying to make it as clean as possible.

Flint is not as powerful as torch, but it is still able to start a fire.

&nbsp;

## Installation

```bash
git clone https://github.com/Renovamen/flint.git
cd flint
python setup.py install
```

or

```bash
pip install git+https://github.com/Renovamen/flint.git --upgrade
```

&nbsp;

## Documentation

Documentation is available [here](https://flint.vercel.app).


&nbsp;

## Example

Add these imports:

```python
import flint
from flint import nn, optim, Tensor
```

Build your net first:

```python
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

Or you may prefer to use a `Sequential` container:

```python
class Net(nn.Module):
    def __init__(self, in_features, n_classes):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features, 5)
            nn.ReLU(),
            nn.Linear(5, n_classes)
        )

    def forward(self, x):
        out = self.model(x)
        return out
```

Define these hyper parameters:

```python
# training parameters
n_epoch = 20
lr = 0.001
batch_size = 5

# model parameters
in_features = 10
out_features = 2
```

Here we generate a fake dataset:

```python
import numpy as np
inputs = np.random.rand(batch_size, in_features)
targets = np.random.randint(0, n_classes, (batch_size, ))
x, y = Tensor(inputs), Tensor(targets)
```

Initialize your model, optimizer and loss function:

```python
net = Net(in_features, n_classes)
optimer = optim.Adam(params=net.parameters(), lr=lr)
loss_function = nn.CrossEntropyLoss()
```

Then we can train it:

```python
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
    correct_preds = flint.eq(preds, y).sum().data
    accuracy = correct_preds / y.shape[0]

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

### Autograd

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
- [x] View
- [x] Transpose
- [x] Permute
- [x] Squeeze
- [x] Unsqueeze
- [x] Padding

### Layers

- [x] Linear
- [x] 1D / 2D Convolution
- [ ] Flatten
- [x] 1D / 2D MaxPooling
- [ ] Dropout
- [ ] BatchNormalization
- [ ] RNN
- [x] Sequential

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


&nbsp;

## License

[MIT](LICENSE)


&nbsp;

## Acknowledgements

Flint is inspired by the following projects:

- [PyTorch](https://github.com/pytorch/pytorch)
- [karpathy/micrograd](https://github.com/karpathy/micrograd)
- [teddykoker/tinyloader](https://github.com/teddykoker/tinyloader)
