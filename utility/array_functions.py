__author__ = 'Aubrey'

import numpy as np
import inspect
import scipy
import sklearn
import matplotlib as plt
import matplotlib.pylab as pl
import copy
from sklearn import preprocessing
def normalize_rows(x):
    return sklearn.preprocessing.normalize(x,axis=1,norm='l1')

def is_invalid(x):
    inds = np.isnan(x) | np.isinf(x)
    return inds

def has_invalid(x):
    return np.any(is_invalid(x))

def replace_invalid(x,x_min=0,x_max=1,allowed=None):
    inds = is_invalid(x)
    if allowed is not None:
        allowed = np.asarray(allowed)
        choices = np.random.choice(len(allowed), inds.shape)
        x[inds] = allowed[choices][inds]
    else:
        x[inds] = np.random.uniform(x_min,x_max,x.shape)[inds]
    return x


def try_toarray(x):
    try:
        x = x.toarray()
    except:
        pass
    return x

def spy(m, prec=.001, size=5):
    pl.spy(m,precision=prec, markersize=size)
    pl.show()

def make_label_matrix(y):
    assert len(y.shape) == 1
    y = y.astype('int32')
    m = y.max()
    n = len(y)
    inds = (np.asarray(range(n)),y)
    o = np.ones(n)
    m = scipy.sparse.csc_matrix((o, inds))
    return m

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

if __name__ == '__main__':
    M = np.ones((3,3))
    M[0,1] = np.nan
    M[0,2] = np.nan
    M2 = replace_invalid(copy.deepcopy(M),10,10)
    M3 = replace_invalid(copy.deepcopy(M),allowed=[3,5,6])
    M4 = normalize_rows(M3)
    pass