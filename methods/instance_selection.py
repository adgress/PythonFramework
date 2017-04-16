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
from sklearn.neighbors import KernelDensity
from sklearn.grid_search import GridSearchCV
from results_class.results import Output
import cvxpy as cvx
import scipy
from data import data as data_lib
from scipy import optimize

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
        configs = deepcopy(self.configs)
        self.target_learner = method.NadarayaWatsonMethod(deepcopy(configs))
        self.target_learner.quiet = True
        self.supervised_loss_func = compute_f_nw

    def compute_kernel(self, x, bandwidth):
        D = array_functions.make_graph_distance(x) / bandwidth
        K = (1 / (x.shape[0] * bandwidth)) * (1 / np.sqrt(2 * np.pi)) * np.exp(-.5 * D ** 2)
        return K

    def compute_density(self, X, sigma):
        K = self.compute_kernel(X, sigma)
        return K.sum(1)

    def compute_predictions(self, X, Y):
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
        #self.f_x = self.compute_predictions(X, Y)
        return super(SupervisedInstanceSelection, self).train_and_test(data)

    def train(self, data):
        opt_data = self.create_opt_data(data)
        X = opt_data.X
        f = create_eval(opt_data)
        method = 'SLSQP'
        constraints = [{
            'type': 'ineq',
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
        print results.x
        print 'done'
        self.learned_distribution = results.x
        self.optimization_value = results.fun
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
    M_inv = np.linalg.inv(XTZX + reg * (subset_size/float(n)) * np.eye(p))
    #M_inv = np.linalg.inv(XTZX + reg * Z.sum() * np.eye(p))
    #M_inv = np.linalg.inv(XTZX + reg * np.eye(p))
    w = M_inv.dot(XTZ.dot(Y))
    #print str((Z*n/subset_size).mean())
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

    def compute_predictions(self, X, Y, Z=None, subset_size=None):
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

def make_uniform_data():
    X = np.linspace(0, 1, 100)
    Y = np.zeros(X.size)
    Y[X < .5] = 0
    Y[X >= .5] = 1
    X = array_functions.vec_to_2d(X)
    return X, Y

def plot_approximation(X, learner, use_scatter=True):
    p_x = learner.p_x
    f_x = learner.f_x
    p_s = learner.p_s
    f_s = learner.f_s

    print 'sorted learned distriction: '
    print str(np.sort(learner.learned_distribution)[-5:]/learner.learned_distribution.sum())
    if use_scatter:
        pass
    pl.plot(X, np.ones(X.size) / X.size, 'ro', X, learner.learned_distribution/learner.learned_distribution.sum(), 'bo')
    pl.show(block=True)
    pl.plot(X, p_x, 'ro', X, p_s, 'bo')
    pl.show(block=True)
    pl.plot(X, f_x, 'r', X, f_s, 'b')
    pl.show(block=True)
    # array_functions.plot_line_sub((X, X), (p_x, p_s))
    # array_functions.plot_line_sub((X, X), (f_x, f_s))

def test_1d_data():
    X, Y = make_uniform_data()
    data = data_lib.Data(X, Y)

    learner = SupervisedInstanceSelection()
    learner.quiet = False
    learner.train_and_test(data)

    p_x = learner.p_x
    p_s = learner.p_s

    f_x = learner.f_x
    f_s = learner.f_s
    plot_approximation(X, learner)


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


def test_nd_data():
    n = 30
    p = 1
    data, w = make_linear_data(n, p)
    X = data.x
    Y = data.y.copy()
    learner = SupervisedInstanceSelectionLinear()
    learner.quiet = False
    learner.train_and_test(data)

    p_x = learner.p_x
    p_s = learner.p_s

    f_x = learner.f_x
    f_s = learner.f_s

    w_x = learner.solve_w(X, Y, learner.learner_reg, subset_size=n)
    w_s = learner.solve_w(X, Y, learner.learner_reg, learner.learned_distribution, subset_size=learner.subset_size)

    w_err = norm(w_x - w_s) / norm(w_x)
    f_err = norm(f_x - f_s) / norm(f_x)
    p_err = norm(p_x - p_s) / norm(p_x)
    print 'w_err: ' + str(w_err)
    print 'f_err: ' + str(f_err)
    print 'p_err: ' + str(p_err)
    if p == 1:
        plot_approximation(X, learner)
    print ''

if __name__ == '__main__':
    #test_1d_data()
    test_nd_data()
    print 'hello'