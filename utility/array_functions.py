__author__ = 'Aubrey'

import numpy as np

def histogram_unique(n):
    bins = np.unique(n)
    return np.histogram(n,np.append(bins,bins.max()+1))[0]

def true(n):
    return np.ones(n, dtype=bool)

def false(n):
    return np.zeros(n, dtype=bool)