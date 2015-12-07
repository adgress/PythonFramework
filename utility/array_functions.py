__author__ = 'Aubrey'

import numpy as np
import inspect
import scipy
import sklearn
from sklearn.metrics import pairwise
import matplotlib as plt
import matplotlib.pylab as pl
import copy
from sklearn import preprocessing
from timer.timer import Timer
from timer.timer import tic
from timer.timer import toc


def vec_to_2d(x):
    return np.reshape(x,(len(x),1))

#code from http://stackoverflow.com/questions/9185768/inverting-permutations-in-python
def inv_permutation(perm):
    inverse = [0] * len(perm)
    for i, p in enumerate(perm):
        inverse[p] = i
    return np.asarray(inverse)

def to_boolean(x, length=None):
    assert x.ndim == 1
    if not length:
        length = max(x.size, x.max())
    b = false(length)
    b[x] = True
    return b

def make_laplacian_kNN(x,k,metric):
    dists = pairwise.pairwise_distances(x,x,metric)
    dists[np.diag(true(x.shape[0]))] = np.inf
    inds = dists.argsort()
    to_keep = inds[:,:k]
    W = np.zeros(dists.shape)
    W_inds = np.asarray(range(x.shape[0]))
    for i in range(k):
        W[W_inds,to_keep[:,i]] = dists[W_inds,to_keep[:,i]]
    D = W.sum(1)
    L = np.diag(D) - W
    L = scipy.sparse.csc_matrix(L)
    return L

def make_laplacian_uniform(x,radius,metric):
    W = pairwise.pairwise_distances(x,x,metric)
    inds = W > radius
    W[inds] = 0
    W[~inds] = 1
    W[np.diag(true(x.shape[0]))] = 0
    W = W / (x.shape[0]*radius)
    D = W.sum(1)
    L = np.diag(D) - W
    return L

def make_laplacian(x, sigma, metric='euclidean'):
    assert False, 'Normalize by 1/sigma?'
    W = make_rbf(x,sigma,metric)
    D = W.sum(1)
    L = np.diag(D) - W
    return L

def make_rbf(x,sigma,metric='euclidean'):
    if metric == 'cosine':
        #This code may be faster for some matrices
        # Code from http://stackoverflow.com/questions/17627219/whats-the-fastest-way-in-python-to-calculate-cosine-similarity-given-sparse-mat
        '''
        tic()
        #x = x.toarray()
        #similarity = np.dot(x, x.T)
        similarity = (x.dot(x.T)).toarray()
        square_mag = np.diag(similarity)
        inv_square_mag = 1 / square_mag
        inv_square_mag[np.isinf(inv_square_mag)] = 0
        inv_mag = np.sqrt(inv_square_mag)
        W = similarity * inv_mag
        W = W.T * inv_mag
        W = 1 - W
        toc()
        tic()
        W2 = pairwise.pairwise_distances(x,x,metric)
        toc()
        '''
        W = pairwise.pairwise_distances(x,x,metric)
    else:
        #tic()
        W = pairwise.pairwise_distances(x,x,metric)
        #toc()
    W = np.square(W)
    W = -sigma * W
    W = np.exp(W)
    return W

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

def move_fig(fig):
    manager = fig.canvas.manager
    w = manager.canvas.width()
    h = manager.canvas.height()
    manager.window.setGeometry(2000,200,w,h)


def plot_2d(x,y,data_set_ids=None):
    pl.close()
    fig = pl.figure(1)
    if data_set_ids is None:
        data_set_ids = np.zeros(y.size)
    pl.scatter(x,y,alpha=.8,c=data_set_ids,s=60)
    manager = pl.get_current_fig_manager()
    #manager.window.SetPosition((1500, 500))
    #fig.canvas.manager.move(1000)
    move_fig(fig)
    pl.show()
    pass

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