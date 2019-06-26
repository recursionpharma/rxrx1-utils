from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

def listlike(v):
    return isinstance(v, (np.ndarray, list, tuple))


def wrap(v):
    if listlike(v):
        return v
    else:
        return [v]


def select_keys(dictionary, keys):
    return {k: dictionary[k] for k in keys}
