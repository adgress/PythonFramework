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
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
from methods import density

import matplotlib as plt
import matplotlib.pylab as pl

bandwidths = np.logspace(-4,4,200)

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
        self.pca = None
        self.instances_to_keep = None

def create_eval(opt_data):
    return lambda x: eval(x, opt_data, return_losses=False)

def compute_p(Z, opt_data):
    Z_diag = np.diag(Z)
    #WZ_density = opt_data.W_density.dot(Z_diag) / Z_diag.sum()
    WZ_density = opt_data.W_density.dot(Z_diag) / opt_data.subset_size
    WZ_density[np.diag_indices_from(WZ_density)] = 0
    p_s = WZ_density.sum(1)
    return p_s

def compute_f_nw(Z, opt_data):
    Z_diag = diag(Z)
    WZ_learner = opt_data.W_learner.dot(Z_diag)
    WZ_learner[np.diag_indices_from(WZ_learner)] = 0
    WZ_Y = WZ_learner.dot(opt_data.Y)
    D = 1 / WZ_learner.sum(1)
    f_s = D * WZ_Y
    return f_s

def eval(Z, opt_data, return_losses=False):
    f_s = opt_data.compute_f(Z, opt_data)
    residual = f_s - opt_data.f_target
    loss_f = norm(residual)**2

    p_s = compute_p(Z, opt_data)
    loss_p = norm(p_s - opt_data.p_target)**2

    n = Z.size
    val = loss_f/n + opt_data.mixture_reg*loss_p/n
    assert np.isfinite(val)
    if return_losses:
        return val, loss_f/n, opt_data.mixture_reg*loss_p/n
    return val


class SupervisedInstanceSelection(method.Method):
    def __init__(self, configs=MethodConfigs()):
        super(SupervisedInstanceSelection, self).__init__(configs)
        self.cv_params = dict()
        self.is_classifier = False
        self.p_s = None
        self.p_x = None
        self.f_s = None
        self.f_x = None
        self.f_x_estimate = None
        self.learned_distribution = None
        self.optimization_value = None
        self.use_linear = False
        self.quiet = False

        self.selected_data = None
        self.full_data = None

        configs = deepcopy(self.configs)
        self.target_learner = method.NadarayaWatsonMethod(deepcopy(configs))
        self.target_learner.configs.use_validation = True
        self.target_learner.configs.results_features = ['y', 'true_y']
        self.target_learner.quiet = True
        self.subset_learner = deepcopy(self.target_learner)
        self.mixture_reg = 1
        self.subset_size = 10
        self.density_reg = .1
        self.num_samples = 5
        self.subset_density_reg = .1
        self.learner_reg = 100
        self.pca = None
        self.output = None

        self.no_f_x = False

        self.is_noisy = None

        if self.use_linear:
            self.supervised_loss_func = compute_f_linear
        else:
            self.supervised_loss_func = compute_f_nw
            self.cv_params['subset_size'] = self.create_cv_params(-5, 5)

    def solve_w(self, X, Y, reg, Z=None, subset_size=None):
        if subset_size is None:
            subset_size = self.subset_size
        if Z is None:
            Z = np.ones(X.shape[0])/X.shape[0]
        w = solve_w(X, Y, reg, Z, subset_size)
        return w

    def compute_predictions(self, X, Y, Z=None, subset_size=None, estimate=False, learner_reg=None):
        if self.use_linear:
            return self.compute_predictions_linear(X, Y, Z, subset_size, estimate, learner_reg)
        else:
            return self.compute_predictions_nonparametric(X, Y, estimate, learner_reg)

    def compute_predictions_linear(self, X, Y, Z=None, subset_size=None, estimate=False, learner_reg=None):
        if learner_reg is None:
            learner_reg = self.learner_reg
        if not estimate:
            return Y.copy()
        if subset_size is None:
            assert Z is None
            subset_size = 1
        w = self.solve_w(X, Y, learner_reg, Z, subset_size)
        return X.dot(w)

    def compute_predictions_nonparametric(self, X, Y, estimate=False, learner_reg=None):
        if learner_reg is None:
            learner_reg = self.learner_reg
        if not estimate:
            return Y.copy()
        W = self.target_learner.compute_kernel(X, X, bandwidth=learner_reg)
        #np.fill_diagonal(W, 0)
        S = array_functions.make_smoothing_matrix(W)
        return S.dot(Y)

    def create_opt_data(self, data, data_test=None):
        I = data.is_labeled & data.is_train
        X = data.x[I]
        Y = data.y[I]
        instances_to_keep = getattr(data, 'instances_to_keep', None)
        #W_learner = self.target_learner.compute_kernel(X, X, self.learner_reg)
        if data_test is None:
            W_learner = density.compute_kernel(X, X, self.learner_reg)
        else:
            W_learner = density.compute_kernel(X, data_test.x, self.learner_reg)
        X_density = X
        if self.pca is not None:
            X_density = self.pca.transform(X_density)

        # Multiple by X.shape[0] because the actual normalization coefficient will be the sum of the learned distribution
        #W_density = X.shape[0] * density.compute_kernel(X_density, None, self.subset_density_reg)
        ratio = float(X.shape[0]) / self.subset_size
        W_density = ratio * density.compute_kernel(X_density, None, self.subset_density_reg)

        opt_data = OptData(
            X, Y,
            self.f_x[I], self.p_x[I],
            W_learner, W_density,
            self.learner_reg, self.density_reg, self.mixture_reg,
            self.supervised_loss_func, self.subset_size
        )
        opt_data.instances_to_keep = instances_to_keep
        return opt_data

    def train_and_test(self, data):
        #data = data.get_subset(data.is_train & data.is_labeled)
        #data.is_regression = True

        self.target_learner.train_and_test(data)
        self.f_x = self.target_learner.predict(data).y
        self.p_x = density.tune_and_predict_density(data.x, data.x, bandwidths)

        ret_val = super(SupervisedInstanceSelection, self).train_and_test(data)
        return ret_val

    def optimize(self, opt_data):
        return self.optimize_nonparametric(opt_data)

    def optimize_nonparametric(self, opt_data):
        assert opt_data.instances_to_keep is None, 'Not implemented yet!'
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
        if not self.running_cv:
            print 'done with cv'
        #print results.x
        #print 'done'
        self.learned_distribution = results.x.copy()
        if not self.running_cv:
            print 'median z: ' + str(np.median(self.learned_distribution / self.learned_distribution.sum()))
        idx = np.argsort(self.learned_distribution)[::-1]
        #self.learned_distribution[idx[:opt_data.subset_size]] = 1
        #self.learned_distribution[idx[opt_data.subset_size:]] = 0
        self.learned_distribution[idx[self.num_samples:]] = 0
        self.learned_distribution[self.learned_distribution > 0] = 1
        self.learned_distribution /= self.learned_distribution.sum()
        self.learned_distribution *= self.num_samples
        _, loss_f, loss_p = eval(self.subset_size * self.learned_distribution / self.learned_distribution.sum(), opt_data, return_losses=True)
        '''
        print results.fun
        print str(self.subset_density_reg) + ' p err: ' + str(loss_p)
        print str(self.learner_reg) + ' f err: ' + str(loss_f)
        '''
        #self.optimization_value = results.fun
        a = loss_f + loss_p
        self.optimization_value = a

    def train(self, data):
        data.is_regression = True
        opt_data = self.create_opt_data(data)
        self.optimize(opt_data)
        l = np.zeros(data.n)
        v = data.is_train & data.is_labeled
        l[v] = self.learned_distribution
        self.learned_distribution = l
        I = self.learned_distribution > 0

        self.selected_data = deepcopy(data)
        self.selected_data.y[~I] = np.nan
        self.selected_data.is_train[:] = True
        self.subset_learner.train_and_test(self.selected_data)

        self.full_data = deepcopy(data)
        #self.full_data.is_train[:] = True
        #self.full_data.y = self.full_data.true_y
        assert(self.full_data.is_train.all())
        assert(self.full_data.is_labeled.all())
        self.target_learner.train_and_test(self.full_data)
        self.is_noisy = data.is_noisy
        self.y_orig = data.y_orig



    def predict_test(self, data_train, data_test):
        opt_data = self.create_opt_data(data_train, data_test)
        return self.supervised_loss_func(self.learned_distribution, opt_data)


    def predict(self, data):
        I = self.selected_data.is_labeled
        #self.f_x = self.target_learner.predict(data).y
        self.f_x = data.true_y
        self.p_x = density.tune_and_predict_density(self.full_data.x, data.x, bandwidths)
        self.f_s = self.subset_learner.predict(data).y
        self.p_s = density.tune_and_predict_density(self.selected_data.x[I], data.x, bandwidths)
        self.var_x = np.abs(data.true_y - self.target_learner.predict(data).y)
        self.var_s = np.abs(data.true_y - self.f_s)

        res_p = np.abs(self.p_x - self.p_s) / np.linalg.norm(self.p_x)
        '''
        if self.use_var:
            res_var = np.abs(self.var_x - self.var_s) / np.linalg.norm(self.var_x)
            res_total = res_p + res_var
        else:
            res_f = np.abs(self.f_x - self.f_s) / np.linalg.norm(self.f_x)
            res_total = res_f + res_p
        '''
        res_f = np.abs(self.f_x - self.f_s) / np.linalg.norm(self.f_x)
        res_total = res_f + res_p
        self.res_total = res_total
        o = Output(data)
        o.res_total = res_total
        o.true_p = self.p_x
        o.p = self.p_s
        o.true_y = self.f_x
        o.y = self.f_s
        o.var_x = self.var_x
        o.var_s = self.var_s
        o.optimization_value = self.optimization_value/data.n
        o.is_noisy = self.is_noisy
        o.is_selected = self.learned_distribution > 0
        o.y_orig = self.y_orig
        o.x_orig = getattr(data, 'x_orig', None)
        self.output = o
        return o

    def get_shared_suffix(self):
        s = ''
        if getattr(self, 'no_f_x'):
            s += '-just_px'
        return s

    @property
    def prefix(self):
        s = 'SupervisedInstanceSelection'
        s += self.get_shared_suffix()
        if self.use_linear:
            s += '-linear'
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


class SupervisedInstanceSelectionGreedy(SupervisedInstanceSelection):
    def __init__(self, configs=MethodConfigs()):
        super(SupervisedInstanceSelectionGreedy, self).__init__(configs)
        self.cv_params = {
            'sigma_p': np.logspace(-3, 3, 10),
            'sigma_y': self.create_cv_params(-5, 5),
            'C': self.create_cv_params(-3, 3),
        }
        self.C = 0
        self.sigma_y = 1
        self.sigma_p = 1
        self.no_f_x = getattr(configs, 'no_f_x', False)
        #self.fixed_sigma_x = getattr(configs, 'fixed_sigma_x', False)
        #self.no_spectral_kernel = getattr(configs, 'no_spectral_kernel', False)
        self.use_p_x = getattr(configs, 'use_p_x', True)

        # Just use p(x)
        if self.no_f_x:
            del self.cv_params['sigma_y']
            del self.cv_params['C']
            self.C = 1

        # Just use f(x)
        if not self.use_p_x:
            #self.cv_params['sigma_p'] = np.asarray([1], dtype=np.float)
            del self.cv_params['sigma_p']
            if 'C' in self.cv_params:
                del self.cv_params['C']

    def evaluate_selection(self, W_p, W_y, I, y_true, p_true):
        p_pred = W_p[:, I].sum(1)
        S = array_functions.make_smoothing_matrix(W_y[:, I])
        y_pred = S.dot(y_true[I])
        error = norm(y_pred - y_true) + self.C * norm(p_pred - p_true)
        return error

    def optimize(self, opt_data):
        #self.sigma_p = 1
        #self.sigma_y = 1
        #self.C = 1
        assert (opt_data.instances_to_keep is None or
                opt_data.instances_to_keep.sum() == 0), 'Not implemented yet!'
        W_p = density.compute_kernel(opt_data.X, None, self.sigma_p)
        W_y = array_functions.make_rbf(opt_data.X, self.sigma_y)
        n = W_p.shape[0]
        selected = array_functions.false(n)
        y_true = self.f_x
        p_true = self.p_x
        for i in range(opt_data.subset_size):
            new_scores = np.zeros(n)
            new_scores[:] = np.inf
            for j in range(n):
                if selected[j]:
                    continue
                b = array_functions.false(n)
                b[j] = True
                new_scores[j] = self.evaluate_selection(W_p, W_y, b | selected, y_true, p_true)
            best_idx = new_scores.argmin()
            selected[best_idx] = True

        self.selected = selected
        if selected.sum() < opt_data.subset_size:
            # print 'Empty clusters'
            pass
        # self.learned_distribution = compute_p(selected, opt_data)
        self.learned_distribution = selected
        self.optimization_value = 0

        #return self.optimize_nonparametric(opt_data)

    @property
    def prefix(self):
        s = 'SupervisedInstanceSelectionGreedy'
        s += self.get_shared_suffix()
        return s

class SupervisedInstanceSelectionCluster(SupervisedInstanceSelection):
    def __init__(self, configs=MethodConfigs()):
        super(SupervisedInstanceSelectionCluster, self).__init__(configs)
        self.mixture_reg = None
        self.cv_params = {}
        self.k_means = KMeans()
        self.original_cluster_inds = None

    def cluster(self, X, num_clusters, k_means=None):
        if k_means is None:
            k_means = KMeans()
        k_means.set_params(n_clusters=num_clusters)
        X_cluster_space = k_means.fit_transform(X)
        I = k_means.predict(X)
        return k_means, X_cluster_space, I

    def optimize(self, opt_data):
        assert opt_data.instances_to_keep is None, 'Not implemented yet!'
        self.k_means, X_cluster_space, _ = \
            self.cluster(opt_data.X, opt_data.subset_size, self.k_means)
        selected, selected_indices = self.compute_centroids_for_clustering(opt_data.X, self.k_means)
        assert selected.sum() == opt_data.subset_size
        #self.learned_distribution = compute_p(selected, opt_data)
        self.learned_distribution = selected
        self.optimization_value = 0

    def compute_data_set_centroid(self, X):
        k_means, X_cluster_space, I = self.cluster(X, 1)
        return self.compute_centroids_for_clustering(X, k_means)


    def compute_centroids_for_clustering(self, X, k_means):
        X_cluster_space = k_means.transform(X)
        centroids = np.zeros(X.shape[0])
        indices = np.zeros(k_means.n_clusters)
        for i in range(k_means.n_clusters):
            xi = X_cluster_space[:, i]
            idx = np.argmin(xi)
            centroids[idx] = 1
            indices[i] = idx
        return centroids, indices.astype(np.int)

    def compute_purity(self, cluster_inds, Y, num_clusters):
        avg_variance = 0
        stds = np.zeros(num_clusters)
        percs = np.zeros(num_clusters)
        means = np.zeros(num_clusters)
        for i in range(num_clusters):
            I = cluster_inds == i
            yi = Y[I]
            means[i] = yi.mean()
            percs[i] = I.mean()
            stds[i] = yi.std()
            avg_variance += I.mean() * yi.std()
        return percs, means, stds

    @property
    def prefix(self):
        s = 'SupervisedInstanceSelectionCluster'
        s += self.get_shared_suffix()
        if self.use_linear:
            s += '-linear'
        return s

class SupervisedInstanceSelectionClusterGraph(SupervisedInstanceSelectionCluster):
    def __init__(self, configs=MethodConfigs()):
        super(SupervisedInstanceSelectionClusterGraph, self).__init__(configs)
        self.mixture_reg = None
        self.cv_params = {
            'sigma_x': self.create_cv_params(-5, 5),
            'sigma_y': self.create_cv_params(-5, 5),
        }
        self.spectral_cluster = SpectralClustering()
        self.original_cluster_inds = None
        self.configs.use_saved_cv_output = True
        self.no_f_x = getattr(configs, 'no_f_x', False)
        self.fixed_sigma_x = getattr(configs, 'fixed_sigma_x', False)
        self.no_spectral_kernel = getattr(configs, 'no_spectral_kernel', False)
        if self.no_f_x:
            del self.cv_params['sigma_y']
        if self.fixed_sigma_x or self.no_spectral_kernel:
            self.cv_params['sigma_x'] = np.asarray([1], dtype=np.float)

    def cluster_spectral(self, W, num_clusters, spectral_cluster=None):
        if spectral_cluster is None:
            spectral_cluster = SpectralClustering()
        spectral_cluster.set_params(
            n_clusters=num_clusters,
            affinity='precomputed',
            n_init=10
        )
        try:
            I = spectral_cluster.fit_predict(W)
        except:
            print 'Spectral clustering failed, clustering on identity matrix'
            try:
                #I = spectral_cluster.fit_predict(np.eye(W.shape[0]))
                I = np.random.choice(range(0, num_clusters), W.shape[0])
            except:
                I = np.random.choice(range(0, num_clusters), W.shape[0])

        return spectral_cluster, I

    def compute_data_set_centroid_spectral(self, W):
        d = W.sum(1)
        return np.argmax(d)

    def compute_centroids_for_spectral_clustering(self, W, cluster_inds):
        v = np.unique(cluster_inds)
        centroid_inds = np.zeros(v.size)
        for i, vi in enumerate(v):
            I = (cluster_inds == vi).nonzero()[0]
            Wi = W[I, :]
            Wi = Wi[:, I]
            a = self.compute_data_set_centroid_spectral(Wi)
            centroid_inds[i] = I[a]
        centroid_inds = centroid_inds.astype(np.int)

        return array_functions.make_vec_binary(centroid_inds, W.shape[0])

    def optimize(self, opt_data):
        instances_to_keep = getattr(opt_data, 'instances_to_keep', None)
        if self.no_spectral_kernel:
            W_x = array_functions.make_graph_distance(opt_data.X)
        else:
            W_x = array_functions.make_rbf(opt_data.X, self.sigma_x)
        W = W_x
        if not self.no_f_x:
            W_y = array_functions.make_rbf(opt_data.Y, self.sigma_y)
            W = W_x * W_y

        num_clusters = opt_data.subset_size
        if instances_to_keep is not None:
            num_clusters += instances_to_keep.sum()
        self.spectral_cluster, cluster_inds = \
            self.cluster_spectral(W, num_clusters, self.spectral_cluster)
        '''
        print [self.sigma_x, self.sigma_y]
        array_functions.plot_histogram(cluster_inds, 21)
        from matplotlib import pyplot as plt
        plt.close()
        '''
        if not self.running_cv:
            I = cluster_inds
            _, I2 = \
                self.cluster_spectral(W, num_clusters, self.spectral_cluster)
            print ''

        selected = self.compute_centroids_for_spectral_clustering(W, cluster_inds)

        #If there are instances we have to select
        if instances_to_keep is not None:
            for i in range(num_clusters):
                this_cluster = cluster_inds == i
                selected_this_cluster = selected & this_cluster
                to_keep_this_cluster = instances_to_keep & this_cluster
                has_fixed_instances = to_keep_this_cluster.any()
                if has_fixed_instances:
                    selected[selected_this_cluster] = False
            if selected.sum() > opt_data.subset_size:
                selected[selected.nonzero()[0][opt_data.subset_size:]] = False
            selected[instances_to_keep] = True
        self.W = W
        self.cluster_inds = cluster_inds
        self.selected = selected
        if selected.sum() < opt_data.subset_size:
            print 'Empty clusters'
            pass
        #self.learned_distribution = compute_p(selected, opt_data)
        self.learned_distribution = selected
        self.optimization_value = 0

    @property
    def prefix(self):
        s = 'SupervisedInstanceSelectionClusterGraph'
        s += self.get_shared_suffix()
        if getattr(self, 'fixed_sigma_x'):
            s += '-fixedSigX'
        return s

class SupervisedInstanceSelectionSubmodular(SupervisedInstanceSelection):
    def __init__(self, configs=MethodConfigs()):
        super(SupervisedInstanceSelectionSubmodular, self).__init__(configs)
        self.mixture_reg = None
        self.cv_params = {
            'sigma_x': self.create_cv_params(-5, 5),
            'sigma_y': self.create_cv_params(-5, 5),
            'C': self.create_cv_params(-3, 3),
        }
        self.original_cluster_inds = None
        self.configs.use_saved_cv_output = True
        self.no_f_x = getattr(configs, 'no_f_x', False)
        self.num_class_splits = getattr(configs, 'num_class_splits', None)
        if self.no_f_x:
            del self.cv_params['sigma_y']

    def evaluate_selection(self, W, I):
        v = W[np.ix_(I, ~I)].sum() - self.C*W[np.ix_(I, I)].sum()
        return v

    def optimize_for_data(self, W, num_to_select):
        selected = array_functions.false(W.shape[0])
        for i in range(num_to_select):
            new_scores = np.zeros(W.shape[0])
            new_scores[:] = -np.inf
            for j in range(W.shape[0]):
                if selected[j]:
                    continue
                b = array_functions.false(W.shape[0])
                b[j] = True
                new_scores[j] = self.evaluate_selection(W, selected | b)
            best_idx = new_scores.argmax()
            selected[best_idx] = True
        return selected

    def optimize(self, opt_data):
        assert opt_data.instances_to_keep is None, 'Not implemented yet!'
        W_x = array_functions.make_rbf(opt_data.X, self.sigma_x)
        W = W_x
        if not self.no_f_x:
            W_y = array_functions.make_rbf(opt_data.Y, self.sigma_y)
            W = W_x * W_y
        n = W.shape[0]
        selected = array_functions.false(W.shape[0])
        splits = [array_functions.true(n)]
        num_per_split = [opt_data.subset_size]
        if self.num_class_splits is not None:
            assert self.num_class_splits == 2
            I1 = opt_data.Y <= opt_data.Y.mean()
            splits = [I1, ~I1]
            num_per_split = [opt_data.subset_size/2, opt_data.subset_size/2]
        for split, num in zip(splits, num_per_split):
            W_split = W[np.ix_(split, split)]
            split_selections = self.optimize_for_data(W_split, num)
            split_inds = split.nonzero()[0]
            selected[split_inds[split_selections]] = True

        #selected = self.compute_centroids_for_spectral_clustering(W, cluster_inds)
        self.W = W
        self.selected = selected
        if selected.sum() < opt_data.subset_size:
            #print 'Empty clusters'
            pass
        #self.learned_distribution = compute_p(selected, opt_data)
        self.learned_distribution = selected
        self.optimization_value = 0

    @property
    def prefix(self):
        s = 'SupervisedInstanceSelectionSubmodular'
        s += self.get_shared_suffix()
        if getattr(self, 'num_class_splits', None) is not None:
            s += '-class_splits=' + str(self.num_class_splits)
        return s

class SupervisedInstanceSelectionClusterSplit(SupervisedInstanceSelectionCluster):
    def __init__(self, configs=MethodConfigs()):
        super(SupervisedInstanceSelectionClusterSplit, self).__init__(configs)
        self.mixture_reg = None
        self.cv_params = {}
        self.k_means = KMeans()
        self.max_std = .4
        self.sub_cluster_size = 2
        self.impure_cluster_samples = 2
        self.original_cluster_inds = None

    def compute_centroids(self, opt_data):
        num_clusters = opt_data.subset_size
        X = opt_data.X
        Y = opt_data.Y
        self.k_means, X_cluster_space, cluster_inds = \
            self.cluster(X, num_clusters, self.k_means)
        percs, means, stds = self.compute_purity(
            cluster_inds, Y, opt_data.subset_size)
        is_impure = stds > self.max_std
        curr_centroids, centroid_indices = self.compute_centroids_for_clustering(X, self.k_means)
        sub_kmeans = KMeans(n_clusters=self.sub_cluster_size)
        selected_centroids = np.zeros(curr_centroids.size)
        self.original_cluster_inds = cluster_inds
        for i in range(num_clusters):
            if not is_impure[i]:
                selected_centroids[centroid_indices[i]] = 1
                continue
            I = cluster_inds == i
            I_inds = I.nonzero()[0]
            xi = X[I, :]
            yi = Y[I]
            sub_kmeans, xi_cluster_space, sub_cluster_inds = \
                self.cluster(xi, self.sub_cluster_size, sub_kmeans)
            percs_sub, means_sub, stds_sub = self.compute_purity(sub_cluster_inds, yi, self.sub_cluster_size)
            if (stds_sub <= self.max_std).all():
                print 'good subset clustering!'
                new_centroids, new_indices = self.compute_centroids_for_clustering(xi, sub_kmeans)
                new_centroids_I = I_inds[new_indices]
                curr_centroids[new_centroids_I] = 1
            else:
                assert self.impure_cluster_samples == 2
                I0 = (yi <= yi.mean()).nonzero()[0]
                I1 = (yi > yi.mean()).nonzero()[0]
                centroid0, centroid0_idx = self.compute_data_set_centroid(xi[I0])
                centroid1, centroid1_idx = self.compute_data_set_centroid(xi[I1])
                curr_centroids[I_inds[I0[centroid0_idx[0]]]] = 1
                curr_centroids[I_inds[I1[centroid1_idx[0]]]] = 1
        #print ''
        return curr_centroids == 1



    def optimize(self, opt_data):
        assert opt_data.instances_to_keep is None, 'Not implemented yet!'
        centroids = self.compute_centroids(opt_data)
        selected = np.zeros(opt_data.Y.size)
        selected[centroids] = 1
        self.learned_distribution = selected
        self.optimization_value = 0


    @property
    def prefix(self):
        s = 'SupervisedInstanceSelectionClusterSplit'
        s += self.get_shared_suffix()
        if self.use_linear:
            s += '-linear'
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
    pl.plot(X, np.ones(X.size) / X.size, 'ro', label='Original Samples')
    pl.plot(X, learner.learned_distribution/learner.learned_distribution.sum(), 'bo', label='Subset Samples')
    pl.legend(loc='center right', fontsize=10)
    pl.show(block=True)
    pl.plot(X, p_x, 'ro', label='Original Distribution')
    pl.plot(X, p_s, 'bo', label='Subset Distribution')
    pl.legend(loc='center right', fontsize=10)
    pl.show(block=True)
    pl.plot(X, f_x_estimate, 'r', label='Original Estimate')
    pl.plot(X, f_s, 'b', label='Subset Estimate')
    pl.legend(loc='center right', fontsize=10)
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


def test_cluster_purity(X, Y, X_test=None, Y_test=None, label_names=None, mnist=False):
    num_clusters = 4
    num_dims = None
    if mnist and num_dims is not None:
        pca = PCA(n_components=num_dims)
        X_orig = X
        X_pca = pca.fit_transform(X)
        X = X_pca
        print 'PCA dims=' + str(num_dims)

    k_means = KMeans()
    k_means.set_params(n_clusters=num_clusters)
    print_cluster_purity(k_means, X, Y)

def print_cluster_purity(k_means, X, Y, allow_recurse=True):
    cluster_inds = k_means.fit_predict(X)
    avg_variance = 0
    for i in range(k_means.n_clusters):
        I = cluster_inds == i
        yi = Y[I]
        perc = I.mean()
        print 'Cluster ' + str(i) + ': perc=' + str(perc)
        print 'Mean: ' + str(yi.mean())
        print 'STD: ' + str(yi.std())
        if array_functions.in_range(yi.mean(), .2, .8) and I.mean() > .2 and allow_recurse:
            print 'Splitting cluster'
            k_means_sub = KMeans(n_clusters=4)
            Xi = X[I, :]
            Yi = Y[I]
            array_functions.plot_heatmap(Xi, Yi + .5, subtract_min=False)
            k_means_sub.fit(Xi)
            print_cluster_purity(k_means_sub, Xi, Yi, allow_recurse=False)
        avg_variance += I.mean() * yi.std()
    print 'Average STD: ' + str(avg_variance)

def test_methods(X, Y, X_test=None, Y_test=None, label_names=None, mnist=False):
    #import statsmodels.api as sm
    #from statsmodels.nonparametric import kernel_density, kernels, _kernel_base
    #W = kernel_density.gpke([.1]*X.shape[1], X.T, X, ['c']* X.shape[1], tosum=False)
    #kernel_density.KDEMultivariate(X_pca, var_type=['c'] * X.shape[1])
    # kernels.gaussian()



    should_print_cluster_purity = False
    use_linear = False
    plot_instances = False
    plot_batch = False
    if mnist:
        plot_instances = True
        plot_batch = True
    num_neighbors = 3
    num_samples = 8
    mixture_reg = 0

    pca = PCA(n_components=2)
    X_orig = X
    X_pca = pca.fit_transform(X)
    X = X_pca
    X_test = pca.transform(X_test)
    print 'explained_variance: ' + str(pca.explained_variance_ratio_.sum())


    best_bandwidth, values = density.get_best_bandwidth(X_pca, bandwidths)

    sub_values = np.zeros(bandwidths.size)
    num_splits = 10
    splits = array_functions.create_random_splits(X.shape[0], num_samples, num_splits=num_splits)
    for i in range(splits.shape[0]):
        I = splits[i, :]
        b, v = density.get_best_bandwidth(X_pca[~I, :], bandwidths, X_pca[I, :])
        sub_values += v
    sub_values /= num_splits
    best_sub_idx = np.argmin(sub_values)
    best_sub_bandwidth = bandwidths[best_sub_idx]
    p = density.compute_density(X_pca, None, best_bandwidth)

    sis = SupervisedInstanceSelection()
    sis.use_linear = use_linear
    cluster = SupervisedInstanceSelectionCluster()
    cluster.use_linear = use_linear
    methods = [
        SupervisedInstanceSelectionClusterSplit(),
        SupervisedInstanceSelectionCluster(),
        #SupervisedInstanceSelection(),
    ]
    for i, m in enumerate(methods):
        m.subset_size = num_samples
        m.num_samples = num_samples
        if i == 1:
            m.subset_size = int(2*num_samples)
            m.num_samples = int(2*num_samples)
        if mnist:
            m.learner_reg = best_bandwidth
        m.density_reg = best_bandwidth
        m.subset_density_reg = best_sub_bandwidth
        #m.pca = pca
        m.mixture_reg = mixture_reg

    '''
    from sklearn.model_selection import GridSearchCV
    params = {'bandwidth': np.logspace(-5, 5)}
    kde = KernelDensity()
    grid = GridSearchCV(kde, params)
    grid.fit(X_pca)

    best_param = grid.best_params_['bandwidth']
    kde.set_params(bandwidth=best_param)
    p = kde.fit(X_pca)
    p = np.exp(kde.score_samples(X_pca))
    p2 = density.compute_density(X_pca, best_param)
    '''

    data = data_lib.Data(X, Y)
    p = X.shape[1]
    for learner in methods:
        learner.quiet = False
        learner.train_and_test(deepcopy(data))
        learner.train(deepcopy(data))
        p_x = learner.p_x
        p_s = learner.p_s

        f_x_estimate = learner.f_x_estimate
        f_s = learner.f_s
        print 'learner: ' + learner.prefix
        if use_linear:
            w_x = learner.solve_w(X, Y, learner.learner_reg, subset_size=1)
            w_s = learner.solve_w(X, Y, learner.learner_reg, learner.learned_distribution,
                                  subset_size=learner.learned_distribution.sum())
            if X_test is not None:
                assert False, 'Compute error on training set instead?'
                y_pred = X_test.dot(w_s)
                #y_pred = np.sign(y_pred)
                #y_pred[y_pred <= 0] = 0
                y_pred = np.round(y_pred)
                err = np.abs(y_pred != Y_test)
                print 'mean error: ' + str(err.mean())
                w_err = norm(w_x - w_s) / norm(w_x)
                print 'w_err: ' + str(w_err)
        else:
            if mnist:
                #Compute error on training set
                y_pred = learner.f_s
                y_pred = np.round(y_pred)
                err = np.abs(y_pred != Y)
                print 'mean error: ' + str(err.mean())


        f_err = norm(f_x_estimate - f_s) / norm(f_x_estimate)
        p_err = norm(p_x - p_s) / norm(p_x)

        print 'f_err: ' + str(f_err)
        print 'p_err: ' + str(p_err)
        print 'sorted learned distriction: '
        sorted_distribution = np.sort(learner.learned_distribution)
        inds = np.argsort(learner.learned_distribution)[::-1]
        print str(sorted_distribution[-num_samples:] / learner.learned_distribution.sum())

        if should_print_cluster_purity:
            k_means = KMeans(n_clusters=num_samples)
            k_means.fit(X[learner.learned_distribution > 0, :])
            print_cluster_purity(k_means, X, Y)
        if plot_instances:
            #plot_vec(w_x, [28, 28])
            #plot_vec(w_s, [28, 28])
            if plot_batch:
                distance_matrix = array_functions.make_graph_distance(X)
                to_plot = np.zeros((num_samples, num_neighbors+1, 28*28))
                labels = np.zeros(to_plot.shape[0:2])
                labels = labels.astype('string')
            for sample_idx in range(num_samples):
                curr_idx = inds[sample_idx]
                xi = X_orig[curr_idx]
                if plot_batch:
                    to_plot[sample_idx, 0, :] = xi
                    d = distance_matrix[curr_idx]
                    closest_inds = np.argsort(d)[1:num_neighbors+1]
                    #labels[sample_idx, 0] = data.y[curr_idx]
                    label_str = label_names[int(data.y[curr_idx])]
                    if learner.original_cluster_inds is not None:
                        label_str += ' (' + str(learner.original_cluster_inds[curr_idx]) + ')'
                    labels[sample_idx, 0] = label_str


                    for neighbor_idx, ind in enumerate(closest_inds):
                        xj = X_orig[ind, :]
                        yj = data.y[ind]
                        to_plot[sample_idx, neighbor_idx+1, :] = xj
                        #labels[sample_idx, neighbor_idx+1] = yj
                        labels[sample_idx, neighbor_idx + 1] = label_names[int(yj)]
                else:
                    plot_vec(xi, [28, 28])
            if plot_batch:
                plot_tensor(to_plot, [28, 28], labels)
        if p == 1:
            plot_approximation(X, learner)
    print ''

def plot_tensor(v, size=None, labels=None):
    fig = pl.figure()

    num_rows = v.shape[0]
    num_cols = v.shape[1]
    pl.subplot(num_rows, num_cols, 1)
    for row in range(num_rows):
        for col in range(num_cols):
            pl.subplot(num_rows, num_cols, row*num_cols + col + 1)
            #pl.title(str(labels[row, col]))
            pl.ylabel(str(labels[row, col]))
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
    n = 100
    p = 20
    data, w = make_linear_data(n, p)
    X = data.x
    Y = data.y.copy()
    test_methods(X, Y)

def add_label_noise_cluster(data, num_neighbors=10):
    idx = np.random.choice(data.n)
    y = data.y[idx]
    d = array_functions.make_graph_distance(data.x)[idx]
    sorted_inds = np.argsort(d)
    data.y[sorted_inds[:num_neighbors]] = 1 - y
    data.true_y[sorted_inds[:num_neighbors]] = 1 - y
    return data

def add_label_noise(data, num_noise=30):
    inds = np.random.choice(data.n, num_noise, replace=False)
    data.y[inds] = 1 - data.true_y[inds]
    data.true_y[inds] = 1 - data.true_y[inds]
    return data



from utility import helper_functions
def test_mnist():
    num_per_class = 50
    data = helper_functions.load_object('../data_sets/mnist/raw_data.pkl')
    classes_to_use = [0, 4, 8, 7]
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
    data.change_labels([classes_to_use[0], classes_to_use[2]], [0, 1])
    data_test = data.get_subset(~to_keep)
    data = data.get_subset(to_keep)
    label_names = [
        str(classes_to_use[0]) + '+' + str(classes_to_use[1]),
        str(classes_to_use[2]) + '+' + str(classes_to_use[3]),
    ]

    #data = add_label_noise_cluster(data, num_neighbors=20)
    #data = add_label_noise(data, 20)
    test_methods(data.x, data.y, data_test.x, data_test.y, label_names, mnist=True)
    #test_cluster_purity(data.x, data.y, data_test.x, data_test.y, label_names, mnist=True)

if __name__ == '__main__':
    #test_1d_data()
    #test_nd_data()
    test_mnist()