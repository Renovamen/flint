from utils import get_accuracy

def test(test_loader, net):
    net.eval()

    for i, batch in enumerate(tqdm(test_loader, desc = 'Testing')):
        images, labels = batch
        scores = net(images)
        # compute accuracy
        accuracy = get_accuracy(scores, labels)

    print('\n * TEST ACCURACY - %.1f percent\n' % (accuracy * 100))
