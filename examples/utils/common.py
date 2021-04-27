import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import flint
from flint import Tensor

def get_accuracy(scores, labels):
    preds = scores.argmax(dim=1)
    correct_preds = flint.eq(preds, labels).sum().data
    accuracy = correct_preds / labels.shape[0]
    return accuracy
