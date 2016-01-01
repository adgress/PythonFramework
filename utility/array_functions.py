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
from sklearn import manifold

def bin_data(x, num_bins=10):
    x = np.squeeze(x)
    assert x.ndim == 1
    counts,bins = np.histogram(x,bins=num_bins)
    return np.digitize(x,bins)

def in_range(x, low, high):
    if x.ndim == 2:
        assert x.shape[1] == 1
    return np.squeeze((x>= low) & (x<=high))

def is_all_zero_row(x):
    return ~x.any(axis=1)

def is_all_zero_column(x):
    return ~x.any(axis=0)

def relative_error(x,y):
    return np.linalg.norm(x-y)/np.linalg.norm(x)

def add_bias(x):
    n = x.shape[0]
    return np.hstack((np.ones((n, 1)), x))

def normalize(x):
    x_min = x.min()
    x_max = x.max()
    delta = x_max-x_min
    if delta == 0:
        delta = 1
    x = (x - x_min).astype('float') / delta
    return x

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
        #W[W_inds,to_keep[:,i]] = dists[W_inds,to_keep[:,i]]
        W[W_inds,to_keep[:,i]] = 1

    #Make symmetric for quad_form
    W = .5*(W + W.T)
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
    manager.window.setGeometry(3000,0,w,h)



def plot_MDS(x, y=None, data_set_ids=None):
    if data_set_ids is None:
        data_set_ids = np.ones(x.shape[0])
    pl.close()

    fig = pl.figure()
    #axes = fig.add_subplot(111)
    axes = pl.subplot(111)
    to_use = false(x.shape[0])
    max_to_use = 200
    for i in np.unique(data_set_ids):
        I = (data_set_ids == i).nonzero()[0]
        if len(I) <= max_to_use:
            to_use[I] = True
        else:
            choice = np.random.choice(I, max_to_use, replace=False)
            to_use[choice] = True
    x = x[to_use,:]
    data_set_ids = data_set_ids[to_use]
    if y is not None:
        y = y[to_use]
    x = try_toarray(x)
    I_nonzero = x.sum(1).nonzero()[0]
    x = x[I_nonzero,:]
    data_set_ids = data_set_ids[I_nonzero]
    y = y[I_nonzero]


    '''
    x = np.zeros(x.shape)
    x[data_set_ids==0,0] = 1
    x[data_set_ids==1,1] = 1
    x[data_set_ids==2,2] = 1
    '''
    W = pairwise.pairwise_distances(x,x,'cosine')
    #W = pairwise.pairwise_distances(x,x,'euclidean')
    W = make_rbf(x,sigma=10,metric='cosine')
    W = 1 - W

    #mds = sklearn.manifold.MDS(dissimilarity='precomputed',max_iter=1000,verbose=1,metric=False)
    mds = sklearn.manifold.MDS(dissimilarity='precomputed',max_iter=1000,verbose=1,n_init=4)
    x_mds = mds.fit_transform(W)
    #mds = sklearn.manifold.MDS()
    #x_mds = mds.fit_transform(x)
    colors = ['r','g','b']
    labels = ['s','o']
    for ind,i in enumerate(np.unique(data_set_ids)):
        if ind == 0 and False:
            continue
        if ind == 2 and False:
            continue
        I = data_set_ids == i
        alpha = 1 / float(I.sum())
        alpha = alpha*50
        alpha = .3
        if i == 0:
            alpha = 1
        y_curr = y[I]
        x_curr = x_mds[I,:]
        for ind2, j in enumerate(np.unique(y_curr)):
            I2 = y_curr == j
            axes.scatter(x_curr[I2,0],x_mds[I2,1], alpha=alpha,c=colors[ind],s=100,marker = labels[ind2])
        #axes.scatter(x_mds[I,0],x_mds[I,1], alpha=alpha,c=colors[ind],s=60)
    move_fig(fig)
    pl.autoscale()
    pl.show(block=False)
    pass


def plot_2d_sub(x,y,data_set_ids=None,alpha=1,title=None):
    pl.close()
    fig = pl.figure(4)
    if data_set_ids is None:
        data_set_ids = np.zeros(y.size)
    u = np.unique(data_set_ids)
    fig.suptitle(title)
    for index, val in enumerate(u):
        if index == 0:
            ax1 = pl.subplot(len(u),1,index+1)

        else:
            pl.subplot(len(u),1,index+1,sharex=ax1,sharey=ax1)
        #pl.title(title)
        inds = data_set_ids == val
        inds = inds.squeeze()
        pl.ylabel(str(val))
        pl.scatter(x[inds],y[inds],alpha=alpha,c='r',s=60)
    move_fig(fig)
    pl.show(block=False)
    pass

def plot_2d(x,y,data_set_ids=None,alpha=1,title=None):
    pl.close()
    fig = pl.figure(1)
    if data_set_ids is None:
        data_set_ids = np.zeros(y.size)
    pl.title(title)
    pl.scatter(x,y,alpha=alpha,c=data_set_ids,s=60)
    move_fig(fig)
    pl.show(block=False)
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
    assert len(a.shape) <= 1 or a.shape[1] == 1
    a = np.squeeze(a)
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