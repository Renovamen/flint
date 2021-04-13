# Simple Example

Here is a simple example to show how Flint work.

## Import

Add these imports:

```python
import flint
from flint import nn, optim, Tensor
```

## Hyper Parameters

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

## Network

Build your network just like in PyTorch:

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

Then initialize your model, optimizer and loss function:

```python
net = Net(in_features, n_classes)
optimer = optim.Adam(params=net.parameters(), lr=lr)
loss_function = nn.CrossEntropyLoss()
```

## Fake Data

Here we generate a fake dataset:

```python
import numpy as np
inputs = np.random.rand(batch_size, in_features)
targets = np.random.randint(0, n_classes, (batch_size, ))
x, y = Tensor(inputs), Tensor(targets)
```

## Train

Finally, we can train it:

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
    preds = scores.argmax(dim=1)
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
