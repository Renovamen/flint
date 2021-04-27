from utils import get_accuracy

def train(n_epochs, train_loader, net, optimer, loss_function, print_freq):
    net.train()

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
            accuracy = get_accuracy(scores, labels)

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
