from methods import method
from methods import transfer_methods
from methods import local_transfer_methods
from configs.base_configs import MethodConfigs
from utility import array_functions
import copy
from copy import deepcopy
import numpy as np
from numpy.linalg import norm
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
    def __init__(self, X, Y, f_target, p_target, W_learner, W_density, learner_reg, density_reg, mixture_reg):
        self.X = X
        self.Y = Y
        self.f_target = f_target
        self.p_target = p_target
        self.W_learner = W_learner
        self.W_density  = W_density
        self.learner_reg = learner_reg
        self.density_reg = density_reg
        self.mixture_reg = mixture_reg

def create_eval(opt_data):
    return lambda x: eval(x, opt_data)

def compute_p(Z, opt_data):
    Z_diag = np.diag(Z)
    WZ_density = opt_data.W_density.dot(Z_diag) / Z_diag.sum()
    p_s = WZ_density.sum(1)
    return p_s

def compute_f(Z, opt_data):
    Z_diag = diag(Z)
    WZ_learner = opt_data.W_learner.dot(Z_diag)
    WZ_Y = WZ_learner.dot(opt_data.Y)
    D = 1 / WZ_learner.sum(1)
    f_s = D * WZ_Y
    return f_s

def eval(Z, opt_data):
    f_s = compute_f(Z, opt_data)
    loss_f = norm(f_s - opt_data.f_target)**2

    p_s = compute_p(Z, opt_data)
    loss_p = norm(p_s - opt_data.p_target)

    n = Z.size
    val = loss_f/n + opt_data.mixture_reg*loss_p/n
    assert np.isfinite(val)
    return val




class SupervisedInstanceSelection(method.Method):
    def __init__(self, configs=MethodConfigs()):
        super(SupervisedInstanceSelection, self).__init__(configs)
        self.cv_params = dict()
        self.cv_params['learner_reg'] = self.create_cv_params(-2, 2)
        self.cv_params['density_reg'] = self.create_cv_params(-2, 2)
        self.cv_params['density_reg'] = np.asarray([.1])
        self.target_density_bandwidth = .01
        self.mixture_reg = 1
        self.subset_size = 5
        configs = deepcopy(self.configs)
        self.target_learner = method.NadarayaWatsonMethod(deepcopy(configs))
        self.target_learner.quiet = True

    def compute_kernel(self, x, bandwidth):
        D = array_functions.make_graph_distance(x) / bandwidth
        K = (1 / (x.shape[0] * bandwidth)) * (1 / np.sqrt(2 * np.pi)) * np.exp(-.5 * D ** 2)
        return K


    def train_and_test(self, data):
        self.target_learner.train_and_test(data)
        X = data.x
        Y = data.y

        #Compute "correct" density
        sigma = self.target_density_bandwidth
        K = self.compute_kernel(X, sigma)
        self.p_x = K.sum(1)

        #Compute "correct" prediction
        W = self.target_learner.compute_kernel(data.x, data.x)
        np.fill_diagonal(W, 0)
        S = array_functions.make_smoothing_matrix(W)
        self.f_x = S.dot(Y)
        return super(SupervisedInstanceSelection, self).train_and_test(data)

    def train(self, data):
        I = data.is_labeled & data.is_train
        X = data.x[I]
        Y = data.y[I]
        W_learner = self.target_learner.compute_kernel(X, X, self.learner_reg)

        #Multiple by X.shape[0] because the actual normalization coefficient will be the sum of the learned distribution
        W_density = X.shape[0] * self.compute_kernel(X, self.density_reg)

        opt_data = OptData(
            X, Y,
            self.f_x[I], self.p_x[I],
            W_learner, W_density,
            self.learner_reg, self.density_reg, self.mixture_reg
        )
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
        print results.x
        print 'done'
        self.learned_distribution = results.x
        self.optimization_value = results.fun
        self.p_s = compute_p(self.learned_distribution, opt_data)
        self.f_s = compute_f(self.learned_distribution, opt_data)

    def predict(self, data):
        o = Output(data)
        o.y[:] = self.optimization_value/data.n
        o.fu = o.y
        return o

    @property
    def prefix(self):
        s = 'SupevisedInstanceSelection'
        return s


def make_uniform_data():
    X = np.linspace(0, 1, 100)
    Y = np.zeros(X.size)
    Y[X < .5] = 0
    Y[X >= .5] = 1
    X = array_functions.vec_to_2d(X)
    return X, Y

if __name__ == '__main__':
    X, Y = make_uniform_data()
    data = data_lib.Data(X, Y)

    learner = SupervisedInstanceSelection()
    learner.quiet = False
    learner.train_and_test(data)

    p_x = learner.p_x
    p_s = learner.p_s

    f_x = learner.f_x
    f_s = learner.f_s

    pl.plot(X, np.ones(X.size)/X.size, 'r', X, learner.learned_distribution, 'b')
    pl.show()
    pl.plot(X, p_x, 'r', X, p_s, 'b')
    pl.show()
    pl.plot(X, f_x, 'r', X, f_s, 'b')
    pl.show()
    #array_functions.plot_line_sub((X, X), (p_x, p_s))
    #array_functions.plot_line_sub((X, X), (f_x, f_s))
    print 'hello'