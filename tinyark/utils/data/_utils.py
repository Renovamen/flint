import numpy as np
from tinyark import Tensor

def default_collate(batch):
    '''
    Puts each data field into a tensor with outer dimension batch size.
    '''
    
    elem = batch[0]
    elem_type = type(elem)

    if isinstance(elem, np.ndarray):
        return Tensor(np.stack(batch))
    if isinstance(elem, (int, float)):
        return Tensor(np.array(batch))
    if isinstance(elem, tuple):
        return elem_type(*(default_collate(samples) for samples in zip(*batch)))
