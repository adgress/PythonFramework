from methods import method
from methods import method
from methods import transfer_methods
from methods import local_transfer_methods
from configs.base_configs import MethodConfigs
from utility import array_functions
import copy
from copy import deepcopy
import numpy as np
from numpy.linalg import norm, inv
from numpy import diag
from data import data as data_lib
from sklearn.datasets import fetch_mldata
from results_class.results import Output
import cvxpy as cvx
import scipy
from data import data as data_lib
from scipy import optimize
from sklearn.cluster import KMeans

import matplotlib as plt
import matplotlib.pylab as pl


class OptData(object):
    def __init__(self, X, Y, f_target, p_target, W_learner, W_density, learner_reg, density_reg, mixture_reg, supervised_loss_func, subset_size):
        self.X = X
        self.Y = Y
        self.f_target = f_target
        self.p_target = p_target
        self.W_learner = W_learner
        self.W_density  = W_density
        self.learner_reg = learner_reg
        self.density_reg = density_reg
        self.mixture_reg = mixture_reg
        self.compute_f = supervised_loss_func
        self.subset_size = subset_size

def create_eval(opt_data):
    return lambda x: eval(x, opt_data)

def compute_p(Z, opt_data):
    Z_diag = np.diag(Z)
    #WZ_density = opt_data.W_density.dot(Z_diag) / Z_diag.sum()
    WZ_density = opt_data.W_density.dot(Z_diag) / opt_data.subset_size
    p_s = WZ_density.sum(1)
    return p_s

def compute_f_nw(Z, opt_data):
    Z_diag = diag(Z)
    WZ_learner = opt_data.W_learner.dot(Z_diag)
    WZ_Y = WZ_learner.dot(opt_data.Y)
    D = 1 / WZ_learner.sum(1)
    f_s = D * WZ_Y
    return f_s

def eval(Z, opt_data):
    f_s = opt_data.compute_f(Z, opt_data)
    residual = f_s - opt_data.f_target
    loss_f = norm(residual)**2

    p_s = compute_p(Z, opt_data)
    loss_p = norm(p_s - opt_data.p_target)**2

    n = Z.size
    val = loss_f/n + opt_data.mixture_reg*loss_p/n
    assert np.isfinite(val)
    return val


class SupervisedInstanceSelection(method.Method):
    def __init__(self, configs=MethodConfigs()):
        super(SupervisedInstanceSelection, self).__init__(configs)
        self.cv_params = dict()
        self.cv_params['learner_reg'] = self.create_cv_params(-2, 2)
        #self.cv_params['density_reg'] = self.create_cv_params(-2, 2)
        #self.cv_params['density_reg'] = np.asarray([.1])
        #density for full data set
        self.density_reg = .1
        #density for subset of data set
        self.subset_density_reg = self.density_reg
        #self.target_density_bandwidth = .01
        self.mixture_reg = 1
        self.subset_size = 5
        self.is_classifier = False
        configs = deepcopy(self.configs)
        self.target_learner = method.NadarayaWatsonMethod(deepcopy(configs))
        self.target_learner.quiet = True
        self.supervised_loss_func = compute_f_nw
        self.p_s = None
        self.p_x = None
        self.f_s = None
        self.f_x = None
        self.f_x_estimate = None
        self.learned_distribution = None
        self.optimization_value = None


    def compute_kernel(self, x, bandwidth):
        D = array_functions.make_graph_distance(x) / bandwidth
        K = (1 / (x.shape[0] * bandwidth)) * (1 / np.sqrt(2 * np.pi)) * np.exp(-.5 * D ** 2)
        return K

    def compute_density(self, X, sigma):
        K = self.compute_kernel(X, sigma)
        return K.sum(1)

    def compute_predictions(self, X, Y, estimate=False):
        if not estimate:
            return Y.copy()
        W = self.target_learner.compute_kernel(X, X)
        np.fill_diagonal(W, 0)
        S = array_functions.make_smoothing_matrix(W)
        return S.dot(Y)

    def create_opt_data(self, data):
        I = data.is_labeled & data.is_train
        X = data.x[I]
        Y = data.y[I]
        W_learner = self.target_learner.compute_kernel(X, X, self.learner_reg)

        # Multiple by X.shape[0] because the actual normalization coefficient will be the sum of the learned distribution
        W_density = X.shape[0] * self.compute_kernel(X, self.subset_density_reg)

        opt_data = OptData(
            X, Y,
            self.f_x[I], self.p_x[I],
            W_learner, W_density,
            self.learner_reg, self.density_reg, self.mixture_reg,
            self.supervised_loss_func, self.subset_size
        )
        return opt_data

    def train_and_test(self, data):
        if self.target_learner is not None:
            self.target_learner.train_and_test(data)
        X = data.x
        Y = data.y

        #Compute "correct" density
        #sigma = self.target_density_bandwidth
        sigma = self.density_reg
        self.p_x = self.compute_density(X, sigma)

        #Compute "correct" prediction
        self.f_x = self.compute_predictions(X, Y, subset_size=X.shape[0])

        self.f_x_estimate = self.compute_predictions(X, Y, subset_size=1, estimate=True)

        #self.f_x = self.compute_predictions(X, Y)
        return super(SupervisedInstanceSelection, self).train_and_test(data)

    def optimize(self, opt_data):
        X = opt_data.X
        f = create_eval(opt_data)
        method = 'SLSQP'
        constraints = [{
            'type': 'eq',
            'fun': lambda z: self.subset_size - z.sum()
        }]
        bounds = [(0, None) for i in range(X.shape[0])]
        z0 = self.subset_size * np.ones(X.shape[0]) / X.shape[0]
        options = None
        args = None
        results = optimize.minimize(
            f,
            z0,
            method=method,
            bounds=bounds,
            jac=None,
            options=options,
            constraints=constraints,
        )
        #print results.x
        print 'done'
        self.learned_distribution = results.x
        self.optimization_value = results.fun

    def train(self, data):
        opt_data = self.create_opt_data(data)
        self.optimize(opt_data)
        self.p_s = compute_p(self.learned_distribution, opt_data)
        self.f_s = self.supervised_loss_func(self.learned_distribution, opt_data)

    def predict(self, data):
        o = Output(data)
        o.y[:] = self.optimization_value/data.n
        o.fu = o.y
        return o

    @property
    def prefix(self):
        s = 'SupervisedInstanceSelection'
        return s


#Note: Z.sum() should equal subset_size
def solve_w(X, Y, reg, Z, subset_size):
    n, p = X.shape

    XTZ = X.T * Z
    #XTZ = X.T * (Z * n / Z.sum())
    XTZX = XTZ.dot(X)
    # because loss term is 1/subset_size ||Xw - Y||^2
    #M_inv = np.linalg.inv(XTZX + reg * (subset_size/float(n)) * np.eye(p))
    M_inv = np.linalg.inv(XTZX + reg * subset_size * np.eye(p))
    w = M_inv.dot(XTZ.dot(Y))
    return w

def compute_f_linear(Z, opt_data):
    #Z /= opt_data.subset_size
    X = opt_data.X
    Y = opt_data.Y
    n, p = X.shape
    reg = opt_data.learner_reg
    w = solve_w(X, Y, reg, Z, opt_data.subset_size)
    y_pred = X.dot(w)
    return y_pred




class SupervisedInstanceSelectionLinear(SupervisedInstanceSelection):
    def __init__(self, configs=MethodConfigs()):
        super(SupervisedInstanceSelectionLinear, self).__init__(configs)
        self.cv_params = dict()
        #self.cv_params['learner_reg'] = self.create_cv_params(-2, 2)
        #self.cv_params['density_reg'] = np.asarray([.1])
        #self.target_density_bandwidth = .01
        self.density_reg = .5
        self.subset_density_reg = .5
        self.mixture_reg = 1
        self.learner_reg = 1
        self.subset_size = 5
        configs = deepcopy(self.configs)
        self.target_learner = method.NadarayaWatsonMethod(deepcopy(configs))
        self.target_learner.quiet = True
        self.supervised_loss_func = compute_f_linear

    def solve_w(self, X, Y, reg, Z=None, subset_size=None):
        if subset_size is None:
            subset_size = self.subset_size
        if Z is None:
            Z = np.ones(X.shape[0])/X.shape[0]
        w = solve_w(X, Y, reg, Z, subset_size)
        return w

    def compute_predictions(self, X, Y, Z=None, subset_size=None, estimate=False):
        if not estimate:
            return Y.copy()
        if subset_size is None:
            assert Z is None
            subset_size = 1
        w = self.solve_w(X, Y, self.learner_reg, Z, subset_size)
        return X.dot(w)

    def create_opt_data(self, data):
        #TODO: this creates W_learner, which isn't necessary for linear method
        opt_data = super(SupervisedInstanceSelectionLinear, self).create_opt_data(data)
        opt_data.W_learner = None
        return opt_data

    @property
    def prefix(self):
        s = 'SupervisedInstanceSelectionLinear'
        return s

class SupervisedInstanceSelectionGreedy(SupervisedInstanceSelectionLinear):
    def __init__(self, configs=MethodConfigs()):
        super(SupervisedInstanceSelectionGreedy, self).__init__(configs)
        self.mixture_reg = 1

    def optimize(self, opt_data):
        w = self.solve_w(opt_data.X, opt_data.Y, opt_data.learner_reg)
        y_pred = opt_data.X.dot(w)
        y_diff = np.abs(y_pred - opt_data.Y)
        p_x = self.p_x
        if not np.isfinite(self.mixture_reg):
            total_error = p_x.copy()
        else:
            total_error = -y_diff + opt_data.mixture_reg*p_x
        selected = np.zeros(y_pred.shape)
        assert y_pred.shape >= opt_data.subset_size
        indicies_to_sample = [np.arange(total_error.size)]
        if self.is_classifier:
            assert False, 'TODO'
            classes = np.unique(opt_data.Y)
            indicies_to_sample = []
            for i in classes:
                indicies_to_sample.append(
                    (opt_data.Y == i).nonzero()[0]
                )
        for i in range(opt_data.subset_size):
            idx = np.argmax(total_error)
            selected[idx] = 1
            total_error[idx] = -np.inf
        self.learned_distribution = compute_p(selected, opt_data)
        self.optimization_value = 0

    @property
    def prefix(self):
        s = 'SupervisedInstanceSelectionLinearGreedy'
        return s

class SupervisedInstanceSelectionCluster(SupervisedInstanceSelectionLinear):
    def __init__(self, configs=MethodConfigs()):
        super(SupervisedInstanceSelectionCluster, self).__init__(configs)
        self.mixture_reg = None
        self.k_means = KMeans()

    def optimize(self, opt_data):
        self.k_means.set_params(n_clusters=opt_data.subset_size)
        X_cluster_space = self.k_means.fit_transform(opt_data.X)
        selected = np.zeros(opt_data.Y.size)
        for i in range(opt_data.subset_size):
            xi = X_cluster_space[:, i]
            idx = np.argmin(xi)
            selected[idx] = 1
        assert selected.sum() == opt_data.subset_size
        self.learned_distribution = compute_p(selected, opt_data)
        self.optimization_value = 0

    @property
    def prefix(self):
        s = 'SupervisedInstanceSelectionCluster'
        return s

def make_uniform_data():
    X = np.linspace(0, 1, 100)
    Y = np.zeros(X.size)
    Y[X < .5] = 0
    Y[X >= .5] = 1
    X = array_functions.vec_to_2d(X)
    return X, Y

def plot_approximation(X, learner, use_scatter=True):
    p_x = learner.p_x
    f_x_estimate = learner.f_x_estimate
    p_s = learner.p_s
    f_s = learner.f_s

    if use_scatter:
        pass
    pl.plot(X, np.ones(X.size) / X.size, 'ro', X, learner.learned_distribution/learner.learned_distribution.sum(), 'bo')
    pl.show(block=True)
    pl.plot(X, p_x, 'ro', X, p_s, 'bo')
    pl.show(block=True)
    pl.plot(X, f_x_estimate, 'r', X, f_s, 'b')
    pl.show(block=True)
    # array_functions.plot_line_sub((X, X), (p_x, p_s))
    # array_functions.plot_line_sub((X, X), (f_x, f_s))

def test_1d_data():
    X, Y = make_uniform_data()
    data = data_lib.Data(X, Y)
    test_methods(X, Y)

def make_linear_data(n, p):
    #X = np.random.uniform(0, 1, (n,p))
    X = np.random.normal(0, scale=.1, size=(n,p))
    idx = int(n/2)
    X[:idx, :] += 1
    X[idx:, :] -= 1
    if p == 1:
        X = np.sort(X, 0)
    w = np.random.randn(p)
    w = np.abs(w)
    w[:] = 1
    noise = np.random.normal(0, scale=1, size=n)
    noise[:] = 0
    Y = X.dot(w)
    data = data_lib.Data(X, Y + noise)
    return data, w


def test_methods(X, Y, X_test=None, Y_test=None):
    plot_instances = True
    plot_batch = True
    num_neighbors = 3
    num_samples = 10
    methods = [
        SupervisedInstanceSelectionCluster(),
        SupervisedInstanceSelectionGreedy(),
        SupervisedInstanceSelectionLinear(),
    ]
    methods[0].subset_size = num_samples
    methods[1].subset_size = num_samples
    data = data_lib.Data(X, Y)
    p = X.shape[1]
    for learner in methods:
        learner.quiet = False
        learner.train_and_test(deepcopy(data))

        p_x = learner.p_x
        p_s = learner.p_s

        f_x_estimate = learner.f_x_estimate
        f_s = learner.f_s

        w_x = learner.solve_w(X, Y, learner.learner_reg, subset_size=1)
        w_s = learner.solve_w(X, Y, learner.learner_reg, learner.learned_distribution,
                              subset_size=learner.learned_distribution.sum())

        if X_test is not None:
            y_pred = X_test.dot(w_s)
            y_pred = np.sign(y_pred)
            y_pred[y_pred == 0] = 1
            err = np.abs(y_pred != Y_test)
            print 'mean error: ' + str(err.mean())

        w_err = norm(w_x - w_s) / norm(w_x)
        f_err = norm(f_x_estimate - f_s) / norm(f_x_estimate)
        p_err = norm(p_x - p_s) / norm(p_x)
        print 'learner: ' + learner.prefix
        print 'w_err: ' + str(w_err)
        print 'f_err: ' + str(f_err)
        print 'p_err: ' + str(p_err)
        print 'sorted learned distriction: '
        sorted_distribution = np.sort(learner.learned_distribution)
        inds = np.argsort(learner.learned_distribution)[::-1]
        print str(sorted_distribution[-5:] / learner.learned_distribution.sum())

        if plot_instances:
            #plot_vec(w_x, [28, 28])
            #plot_vec(w_s, [28, 28])
            if plot_batch:
                distance_matrix = array_functions.make_graph_distance(X)
                to_plot = np.zeros((num_samples, num_neighbors+1, 28*28))
            for sample_idx in range(num_samples):
                xi = X[inds[sample_idx]]
                if plot_batch:
                    to_plot[sample_idx, 0, :] = xi
                    d = distance_matrix[inds[sample_idx]]
                    closest_inds = np.argsort(d)[1:num_neighbors+1]
                    for neighbor_idx, ind in enumerate(closest_inds):
                        xj = X[ind, :]
                        to_plot[sample_idx, neighbor_idx+1, :] = xj
                else:
                    plot_vec(xi, [28, 28])
            if plot_batch:
                plot_tensor(to_plot, [28, 28])
        if p == 1:
            plot_approximation(X, learner)
    print ''

def plot_tensor(v, size=None):
    fig = pl.figure()

    num_rows = v.shape[0]
    num_cols = v.shape[1]
    pl.subplot(num_rows, num_cols, 1)
    for row in range(num_rows):
        for col in range(num_cols):
            pl.subplot(num_rows, num_cols, row*num_cols + col + 1)
            x = v[row, col, :]
            plot_vec(x, size, fig, show=False)
    array_functions.move_fig(fig, 1000, 1000, 2500, 100)
    pl.show(block=True)


def plot_vec(v, size=None, fig=None, show=True):
    v = v.copy()
    v += v.min()
    if v.max() != 0:
        v /= v.max()
    if size is not None:
        v = np.reshape(v, size)
    if fig is None:
        fig = pl.figure()
    pl.imshow(v, cmap=pl.cm.gray)
    if fig is None:
        array_functions.move_fig(fig, 500, 500, 2000, 1000)
    if show:
        pl.show(block=True)
    pass

def test_nd_data():
    n = 30
    p = 20
    data, w = make_linear_data(n, p)
    X = data.x
    Y = data.y.copy()
    test_methods(X, Y)


from utility import helper_functions
def test_mnist():
    num_per_class = 30
    data = helper_functions.load_object('../data_sets/mnist/raw_data.pkl')
    classes_to_use = [0, 3, 4, 7]
    I = array_functions.find_set(data.y, classes_to_use)
    data = data.get_subset(I)
    to_keep = None
    for i in classes_to_use:
        inds = (data.y == i).nonzero()[0]
        I = np.random.choice(inds, size=num_per_class, replace=False)
        if to_keep is None:
            to_keep = I
        else:
            to_keep = np.concatenate((to_keep, I))
    data.change_labels([classes_to_use[1], classes_to_use[3]], [classes_to_use[0], classes_to_use[2]])
    data.change_labels([classes_to_use[0], classes_to_use[2]], [-1, 1])
    data_test = data.get_subset(~to_keep)
    data = data.get_subset(to_keep)
    test_methods(data.x, data.y, data_test.x, data_test.y)

if __name__ == '__main__':
    #test_1d_data()
    #test_nd_data()
    test_mnist()
    print 'hello'