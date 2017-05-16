__author__ = 'Aubrey'

import numpy as np
import inspect
import scipy
import sklearn
from sklearn.metrics import pairwise
import matplotlib as plt
import matplotlib.pylab as pl
import copy
from numpy.linalg import norm
from sklearn import preprocessing
from timer.timer import Timer
from timer.timer import tic
from timer.timer import toc
from sklearn import manifold
import warnings
import random
import re
from sklearn.feature_selection import SelectKBest, f_regression
import math
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph


def to_binary_vec(v):
    u = np.unique(v)
    assert u.size <= 2
    x = np.zeros(v.size)
    x[v == u[0]] = 0
    if u.size == 2:
        x[v == u[1]] = 1
    return x


def make_vec_binary(I, n):
    v = false(n)
    v[I] = True
    return v

def create_random_splits(n, num_or_perc_sample, num_splits):
    assert n > 0
    assert num_or_perc_sample > 0
    assert num_splits > 0
    if num_or_perc_sample < 1:
        num_or_perc_sample = math.ceil(num_or_perc_sample * n)
    splits = false((num_splits, n))
    for i in range(num_splits):
        I = np.random.choice(n, num_or_perc_sample, replace=False)
        splits[i, :] = make_vec_binary(I, n)
    return splits


def is_in_percentile(v, p_min, p_max):
    assert np.squeeze(v).ndim == 1
    I = np.argsort(v)
    #v_sorted = v[I]
    i_min = int(math.floor(p_min*v.size))
    i_max = int(math.ceil(p_max*v.size))
    is_in_range = false(v.size)
    is_in_range[I[i_min:i_max]] = True
    return is_in_range


def find_closest(x, val):
    vals = np.zeros(2)
    less_than = x[x <= val]
    if less_than.size == 0:
        vals[0] = np.nan
    else:
        vals[0] = less_than.max()
    greater_than = x[x >= val]
    if greater_than.size == 0:
        vals[1] = np.nan
    else:
        vals[1] = greater_than.min()
    return vals

def select_k_features(x, y, num_features):
    assert num_features <= x.shape[1]
    select_k_best = SelectKBest(f_regression, num_features)
    x = select_k_best.fit_transform(x, y)
    return x


#There are more computationally efficient ways of doing this
def remove_quotes(x):
    for i, elem in enumerate(x):
        for j, e in enumerate(elem):
            elem[j] = re.sub('"', '', e)
        x[i, :] = elem
    return x

def normalized_error(v1, v2):
    return norm(v1/norm(v1) - v2/norm(v2))

def sample_n_tuples(n_or_vector, num_samples=1, tuple_size=2, shuffle_tuples=False):
    if np.asarray(n_or_vector).size == 1:
        n_or_vector = np.asarray(range(n_or_vector))
    p = np.ones(n_or_vector.shape)
    tuples = set()
    draws = 0
    while len(tuples) < num_samples:
        if draws > 10*num_samples:
            warnings.warn('Took too many draws to generate typles')
            break
        draws += 1
        x = np.random.choice(n_or_vector, tuple_size, replace=False)
        x.sort()
        tuples.add(tuple(x))
    if shuffle_tuples:
        shuffled_items = set()
        for x in tuples:
            y = list(x)
            random.shuffle(y)
            shuffled_items.add(tuple(y))
        tuples = shuffled_items
    return tuples

def sample_pairs(n_or_vector, num_samples=1, test_func=None):
    if np.asarray(n_or_vector).size == 1:
        n_or_vector = np.asarray(range(n_or_vector))
    p = np.ones(n_or_vector.shape)
    pairs = set()
    draws = 0
    while len(pairs) < num_samples:
        if draws > 30*num_samples:
            warnings.warn('Took too many draws to generate pairs')
            break
        draws += 1
        x = np.random.choice(n_or_vector, 2, replace=False)
        x.sort()
        if test_func is not None and not test_func(x):
            continue
        pairs.add((x[0], x[1]))
    return pairs

def sample(n_or_vector, num_samples=1, distribution=None):
    try:
        indices = np.random.choice(n_or_vector.shape[0], num_samples, replace=False, p=distribution)
        if n_or_vector.ndim == 1:
            return n_or_vector[indices]
        else:
            return n_or_vector[indices,:]
    except Exception as error:
        assert n_or_vector == int(n_or_vector)
        assert num_samples == int(num_samples)
        return np.random.choice(int(n_or_vector), int(num_samples), replace=False, p=distribution)

def clip(x, min, max):
    assert min <= max
    y = x.copy()
    y[x < min] = min
    y[x > max] = max
    return y

def plot_matrix(x):
    pl.matshow(x, cmap=pl.cm.gray)
    pl.show()

def plot_histogram(x, num_bins=10):
    assert x.ndim == 1 or x.shape[1] == 1
    pl.hist(x, num_bins, normed=1, facecolor='green', alpha=0.75)
    #move_fig()
    pl.show()

def append_rows(x,y):
    if x.ndim == 1:
        x = np.expand_dims(x, 1)
    if y.ndim == 1:
        y = np.expand_dims(y,1)
    z = np.squeeze(np.vstack((x, y)))
    return z

def append_cols(x,y):
    return np.vstack((x, y))

def append_column(x,y):
    return np.hstack((x, vec_to_2d(y)))

def make_smoothing_matrix(W):
    W = replace_invalid(W,0,0)
    D = W.sum(1)
    D[D==0] = 1
    D[~np.isfinite(D)] = 1
    try:
        D_inv = 1 / D
    except:
        D_inv = np.eye(D.shape[0])
    D_inv = replace_invalid(D_inv,x_min=1,x_max=1)
    S = (W.swapaxes(0, 1) * D_inv).swapaxes(0, 1)
    return S

def bin_data(x, num_bins=10):
    #assert False, 'Use np.digitize instead?'
    x = np.squeeze(x)
    assert x.ndim == 1
    bins = np.empty(num_bins-1)
    step = 100.0 / num_bins
    for i in range(num_bins-1):
        p = (i+1)*step
        bins[i] = np.percentile(x,p)
    #counts,bins = np.histogram(x,bins=num_bins)
    return np.digitize(x, bins)

def in_range(x, low, high):
    if x.ndim == 2:
        assert x.shape[1] == 1
    return np.squeeze((x>= low) & (x<=high))

def is_all_zero_row(x):
    return ~x.any(axis=1)

def is_all_zero_column(x):
    return ~x.any(axis=0)

def relative_error(x,y):
    assert x.shape == y.shape
    return np.linalg.norm(x-y)/np.linalg.norm(x)

def add_bias(x):
    n = x.shape[0]
    return np.hstack((np.ones((n, 1)), x))

def standardize(x):
    x = preprocessing.scale(x)
    return x

def normalize(x, min_override=None, max_override=None):
    x = x.copy()
    I = ~np.isfinite(x)
    if (~I).size > 0:
        if min_override is None:
            x_min = x[~I].min()
        else:
            x_min = min_override
        if max_override is None:
            x_max = x[~I].max()
        else:
            x_max = max_override
        delta = x_max-x_min
        if delta == 0:
            delta = 1
        x = (x - x_min).astype('float') / delta
    x[I] = 0
    return x

def vec_to_2d(x):
    if x.ndim == 2:
        return x
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


def woodbury(A_inv, C_inv, U, V):
    VA = V.dot(A_inv)
    B = np.linalg.inv(C_inv + VA.dot(U))
    return A_inv - A_inv.dot(U).dot(B).dot(VA)

#Approximate (lambda I + diag(X.sum()) - X)^-1
def nystrom_woodbury_laplacian(X, lamb, perc_columns, W=None, C=None, D=None, v=None):
    lamb = float(lamb)
    timing_test = False
    if timing_test:
        tic()
    if W is None or C is None:
        W, C = nystrom(X, perc_columns)
    #W_inv = np.linalg.pinv(W)
    #X_n = X.shape[0]
    d = X.sum(1)
    dl_inv = 1/(d+lamb)

    inv_approx = None
    vProd = None
    fast_solver = True
    if fast_solver:
        CTA = C.T*dl_inv
        B_inv = np.linalg.pinv(-W + CTA.dot(C))
        if v is not None:
            assert False, 'Make sure this works'
            v1 = CTA.dot(v)
            v2 = B_inv.dot(v1)
            v3 = -C.dot(v2)
            v4 = v3 + v
            v5 = dl_inv * v4
            vProd = v5
        else:
            T = -C.dot(B_inv).dot(CTA)
            T[np.diag_indices_from(T)] += 1
            inv_approx = dl_inv[:, None] * T
        '''
        vProd = inv_approx.dot(v)
        err = norm(vProd - v5) / norm(vProd)
        print str(err)
        print ''
        '''
    else:
        A_inv = np.diag(1 / (d + lamb))
        CTA = C.T.dot(A_inv)
        B_inv = np.linalg.pinv(-W + CTA.dot(C))
        inv_approx = A_inv - A_inv.dot(C).dot(B_inv).dot(CTA)
    #inv_approx = A_inv.dot(np.eye(A_inv.shape[0]) - C.dot(B_inv).dot(CTA))
    '''

    '''
    #print 'optimized approx error: ' + str(norm(inv_approx-inv_approx2))
    if timing_test:
        toc()
        tic()
        inv_actual = np.linalg.inv(lamb*np.eye(X.shape[0]) + np.diag(d) - X)
        print 'Nystrom-Woodbery error: ' + str(norm(inv_approx-inv_actual)/norm(inv_actual))
        toc()
    return inv_approx, vProd

def nystrom(x, perc_columns):
    timing_test = False
    num_columns = x.shape[1]
    num_sampled_columns = int(np.ceil(perc_columns * num_columns))
    sampled_columns = np.random.choice(num_columns, num_sampled_columns, replace=False)
    sampled_columns = np.sort(sampled_columns)
    x = try_toarray(x)
    C = x[:, sampled_columns]
    W = x[sampled_columns, :]
    W = W[:, sampled_columns]
    #W_inv = np.linalg.pinv(W, rcond=1e-5)
    #Lapprox = C.dot(W_inv).dot(C.T)
    #print 'L Error: ' + str(norm(x - Lapprox) / norm(x))
    if timing_test:
        x_approx = C.dot(np.linalg.pinv(W)).dot(C.T)
        print 'Nystrom error: ' + str(norm(x_approx - x)/norm(x))
    return W, C

def make_graph_adjacent(x, metric):
    assert metric == 'euclidean'
    assert x.shape[1] == 1
    I = x.argsort(0)
    x = normalize(x)
    dists = pairwise.pairwise_distances(x,x,metric)
    W = np.zeros((I.size, I.size))
    for i, x_index in enumerate(I):
        if i > 0:
            neighbor_left = I[i-1]
            W[x_index, neighbor_left] = 1
        if i < I.size-1:
            neighbor_right = I[i+1]
            W[x_index, neighbor_right] = 1
    return W


    dists[np.diag(true(x.shape[0]))] = 0
    dists[dists > radius] = 0
    dists[dists != 0] = 1
    return dists

def make_graph_radius(x, radius, metric='euclidean', normalize_dists=True):
    use_sklearn = False
    if use_sklearn:
        dists = radius_neighbors_graph(
            x, radius, mode='connectivity', metric=metric
        )
    else:
        assert metric == 'euclidean'
        p = x.shape[1]
        #max_dist = norm(np.ones(p) - np.zeros(p))
        x = normalize(x)
        #dists = pairwise.pairwise_distances(x,x,metric) / max_dist
        dists = pairwise.pairwise_distances(x, x, metric)
        if normalize_dists:
            dists /= dists.max()
        dists[np.diag_indices_from(dists)] = 0
        dists[dists > radius] = 0
        dists[dists != 0] = 1
    return dists

def find_knn(x, y, k=10, metric='euclidean'):
    W = make_graph_distance(x, y, metric)
    sorted_inds = np.argsort(W, 0)
    return sorted_inds.T[:, :k]


def make_graph_distance(x, x2=None, metric='euclidean'):
    x = vec_to_2d(x)
    if x2 is None:
        x2 = x
    return pairwise.pairwise_distances(x, x2, metric)

def make_laplacian_with_W(W, normalized=False):
    D = W.sum(1)
    L = np.diag(D) - W
    if normalized:
        D_inv2 = np.diag(1/(D ** .5))
        L = D_inv2.dot(L).dot(D_inv2)
    L = scipy.sparse.csc_matrix(L)
    return L

def make_laplacian_kNN(x,k,metric='euclidean'):
    '''
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
    #W = .5*(W + W.T)
    L = make_laplacian_with_W(W)
    return L
    '''
    # assert False, 'Normalize by 1/sigma?'
    W = make_knn(x, k, metric)
    W = .5*(W + W.T)
    D = W.sum(1)
    L = np.diag(D) - W
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
    #assert False, 'Normalize by 1/sigma?'
    W = make_rbf(x,sigma,metric)
    D = W.sum(1)
    L = np.diag(D) - W
    return L

def make_knn(x, k, metric='euclidean', x2=None, normalize_entries=True):
    if x.ndim == 1:
        x = np.expand_dims(x, 1)
    no_x2 = True
    if x2 is None:
        x2 = x
        no_x2 = False
    use_sklearn = True
    if use_sklearn and no_x2:
        Z = kneighbors_graph(
            x,
            k,
            mode='connectivity',
            metric=metric
        )
    else:
        W = pairwise.pairwise_distances(x,x2,metric)
        I = np.argsort(W)
        Z = np.zeros(W.shape)
        n = W.shape[0]
        for i in range(k):
            if i+1 >= I.shape[1]:
                break
            Z[(np.arange(n), I[:, i+1])] = 1
    if normalize_entries and Z.sum() > 0:
        Z /= Z.sum()
    return Z

def make_rbf(x,sigma,metric='euclidean', x2=None):
    if x.ndim == 1:
        x = np.expand_dims(x, 1)
    if x2 is None:
        x2 = x
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
        W = pairwise.pairwise_distances(x,x2,metric)
    else:
        #tic()
        W = pairwise.pairwise_distances(x,x2,metric)
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

def move_fig(fig, width=None, height=None, x=None, y=None):
    if x is None:
        x = 2000
    if y is None:
        y = 100
    manager = fig.canvas.manager
    w = manager.canvas.width()
    h = manager.canvas.height()
    if width is not None:
        w = width
    if height is not None:
        h = height
    manager.window.setGeometry(x,y,w,h)



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
    pl.show(block=True)
    pass

def plot_line(x,y,title=None,y_axes=None,fig=None,show=True):
    plot_line_sub([x],[y],title,y_axes,fig,show)

def plot_line_sub(x_list, y_list, title=None, y_axes = None,fig=None,show=True):
    #pl.close()
    passed_fig = fig is not None
    if fig is None:
        fig = pl.figure(len(x_list))
    if title is not None:
        fig.suptitle(title)
    for i, (x,y) in enumerate(zip(x_list, y_list)):
        if not passed_fig:
            assert len(x_list) == 1
            pl.subplot(len(x_list),1,i)
        if y_axes is not None:
            a = pl.gca()
            a.set_ylim(y_axes)

        pl.plot(x,y)
    if show:
        move_fig(fig)
        pl.show(block=True)

def plot_heatmap(x, y_mat, alpha=1,title=None,sizes=None,share_axis=False, fig=None, subtract_min=True):
    if sizes is None:
        sizes = 60
    if y_mat.ndim == 1:
        y_mat = np.expand_dims(y_mat, 1)
    new_fig = False
    if fig is None:
        new_fig = True
        pl.close()
        fig = pl.figure(4)
    if title is not None:
        fig.suptitle(title)
    finite_y = y_mat[np.isfinite(y_mat[:])]
    y_min = None
    y_max = None
    if not subtract_min:
        y_min = 0
    #y_min = finite_y.min()
    #y_max = finite_y.max()
    #y_mat = np.log(y_mat)
    #y_max = finite_y.max()
    for index, y in enumerate(y_mat.T):
        if index == 0:
            ax1 = pl.subplot(y_mat.shape[1], 1, index + 1)
        else:
            if share_axis:
                pl.subplot(y_mat.shape[1], 1, index + 1, sharex=ax1, sharey=ax1)
            else:
                pl.subplot(y_mat.shape[1], 1, index + 1)
        red_values = normalize(y, min_override=y_min, max_override=y_max)
        I = np.isfinite(y) & np.isfinite(x[:,0]) & np.isfinite(x[:,1])
        colors = np.zeros((red_values.size, 4))
        #colors[:,0] = red_values
        colors[:, 0] = 0
        alpha = red_values
        colors[:,3] = alpha
        pl.ylabel(str(index))
        if I.mean > 0:
            print 'Percent skipped due to nans: ' + str(1-I.mean())
        pl.scatter(x[I,0], x[I,1], c=colors[I,:], edgecolors='none', s=sizes, zorder=1)
    move_fig(fig, 1000, 1000)

    if new_fig:
        pl.show(block=True)
    pass


def plot_2d_sub_multiple_y(x,y_mat,alpha=1,title=None,sizes=None,share_axis=False):
    if sizes is None:
        sizes = 60
    if y_mat.ndim == 1:
        y_mat = np.expand_dims(y_mat, 1)
    pl.close()
    fig = pl.figure(4)
    fig.suptitle(title)
    for index, y in enumerate(y_mat.T):
        if index == 0:
            ax1 = pl.subplot(y_mat.shape[1], 1, index + 1)
        else:
            if share_axis:
                pl.subplot(y_mat.shape[1], 1, index + 1, sharex=ax1, sharey=ax1)
            else:
                pl.subplot(y_mat.shape[1], 1, index + 1)
        pl.ylabel(str(index))
        pl.scatter(x, y, alpha=alpha, c='r', s=sizes)
    move_fig(fig, 1000, 1000)
    pl.show(block=True)
    pass

def plot_2d_sub(x,y,data_set_ids=None,alpha=1,title=None,sizes=None):
    if sizes is None:
        sizes = 60
    pl.close()
    fig = pl.figure(4)
    if data_set_ids is None:
        data_set_ids = np.zeros(y.size)
    u = np.unique(data_set_ids)
    fig.suptitle(title)
    min_x = x.min()
    max_x = x.max()
    for index, val in enumerate(u):
        '''
        if index == 0:
            ax1 = pl.subplot(len(u),1,index+1)

        else:
            pl.subplot(len(u),1,index+1,sharex=ax1,sharey=ax1)
        '''
        ax = pl.subplot(len(u), 1, index + 1)
        #pl.title(title)
        inds = data_set_ids == val
        inds = inds.squeeze()
        pl.ylabel(str(val))
        pl.scatter(x[inds],y[inds],alpha=alpha,c='r',s=sizes,)
        ax.set_xlim([min_x, max_x])
    move_fig(fig)
    pl.show(block=True)
    pass

def plot_2d(x,y,data_set_ids=None,alpha=1,title=None):
    pl.close()
    fig = pl.figure(1)
    if data_set_ids is None:
        data_set_ids = np.zeros(y.size)
    pl.title(title)
    pl.scatter(x,y,alpha=alpha,c=data_set_ids,s=60)
    #move_fig(fig)
    pl.show(block=True)
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
    try:
        len(to_find)
    except:
        to_find = [to_find]
    for i in to_find:
        inds = inds | (i == a)
    return inds

def find_first_element(a, to_find):
    I = np.argwhere(a == to_find)
    if I.size == 0:
        return None
    return I[0][0]

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