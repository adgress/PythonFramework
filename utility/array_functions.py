__author__ = 'Aubrey'

import numpy as np
import inspect
import scipy

def is_matrix(a):
    scipy_classes = tuple(x[1] for x in inspect.getmembers(scipy.sparse,inspect.isclass))
    return a.__class__ in scipy_classes or isinstance(a, np.ndarray)

def find_set(a,to_find):
    assert len(a.shape) <= 1
    inds = false(len(a))
    for i in to_find:
        inds = inds | (i == a)
    return inds


def histogram_unique(n):
    bins = np.unique(n)
    return np.histogram(n,np.append(bins,bins.max()+1))[0]

def true(n):
    return np.ones(n, dtype=bool)

def false(n):
    return np.zeros(n, dtype=bool)