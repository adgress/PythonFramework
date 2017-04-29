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
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
from methods import density

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
        self.pca = None

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

        configs = deepcopy(self.configs)
        self.target_learner = method.NadarayaWatsonMethod(deepcopy(configs))
        self.target_learner.quiet = True
        self.mixture_reg = 1
        self.subset_size = 10
        self.density_reg = .1
        self.num_samples = 5
        self.subset_density_reg = .1
        self.learner_reg = 100
        self.pca = None

        if self.use_linear:
            self.supervised_loss_func = compute_f_linear
        else:
            self.supervised_loss_func = compute_f_nw
            self.cv_params['subset_size'] = self.create_cv_params(-5, 5)
            #self.cv_params['learner_reg'] = self.create_cv_params(-4, 4)
            #self.cv_params['density_reg'] = self.create_cv_params(-2, 2)
            #self.cv_params['subset_density_reg'] = self.create_cv_params(-4, 4)

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
        return opt_data

    def train_and_test(self, data):
        if self.target_learner is not None:
            self.target_learner.train_and_test(data)
        X = data.x
        Y = data.y

        #Compute "correct" density
        #sigma = self.target_density_bandwidth
        sigma = self.density_reg
        X_density = X
        if self.pca is not None:
            X_density = self.pca.transform(X_density)
        self.p_x = density.compute_density(X_density, None, sigma)

        #self.f_x = self.compute_predictions(X, Y, subset_size=X.shape[0])
        self.f_x = self.compute_predictions(
            X, Y,
            subset_size=X.shape[0],
            learner_reg=self.target_learner.sigma,
            estimate=False
        )
        #f1 = self.f_x
        #f2 = self.target_learner.predict(data).y

        #self.f_x_estimate = self.compute_predictions(X, Y, subset_size=1, estimate=True)
        self.f_x_estimate = self.f_x.copy()
        f_x_estimates = self.f_x_estimate
        #self.f_x = self.compute_predictions(X, Y)
        return super(SupervisedInstanceSelection, self).train_and_test(data)

    def optimize(self, opt_data):
        return self.optimize_nonparametric(opt_data)

    def optimize_nonparametric(self, opt_data):
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
        opt_data = self.create_opt_data(data)
        self.optimize(opt_data)
        self.p_s = compute_p(self.subset_size * self.learned_distribution / self.learned_distribution.sum(), opt_data)
        self.f_s = self.supervised_loss_func(self.learned_distribution, opt_data)

    def predict_test(self, data_train, data_test):
        opt_data = self.create_opt_data(data_train, data_test)
        return self.supervised_loss_func(self.learned_distribution, opt_data)


    def predict(self, data):
        o = Output(data)

        o.y[:] = self.optimization_value/data.n
        o.fu = o.y.copy()
        o.true_y[:] = 0
        return o

    @property
    def prefix(self):
        s = 'SupervisedInstanceSelection'
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

    def optimize(self, opt_data):
        y_pred = self.compute_predictions(
            opt_data.X,
            opt_data.Y,
            Z=None,
            subset_size=None,
            estimate=True,
            learner_reg=opt_data.learner_reg
        )
        y_diff = np.abs(y_pred - opt_data.Y)
        '''
        if self.use_linear:
            w = self.solve_w(opt_data.X, opt_data.Y, opt_data.learner_reg)
            y_pred = opt_data.X.dot(w)
            y_diff = np.abs(y_pred - opt_data.Y)
        '''
        p_x = self.p_x
        if not np.isfinite(self.mixture_reg):
            total_error = p_x.copy()
        else:
            total_error = -y_diff + opt_data.mixture_reg*p_x
        selected = np.zeros(y_pred.shape)
        assert y_pred.shape >= opt_data.subset_size
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
        if self.use_linear:
            s += '-linear'
        return s

class SupervisedInstanceSelectionCluster(SupervisedInstanceSelection):
    def __init__(self, configs=MethodConfigs()):
        super(SupervisedInstanceSelectionCluster, self).__init__(configs)
        self.mixture_reg = None
        self.cv_params = {}
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
        #self.learned_distribution = compute_p(selected, opt_data)
        self.learned_distribution = selected
        self.optimization_value = 0

    @property
    def prefix(self):
        s = 'SupervisedInstanceSelectionCluster'
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



    should_print_cluster_purity = True
    use_linear = False
    plot_instances = False
    plot_batch = False
    if mnist:
        plot_instances = True
        plot_batch = True
    num_neighbors = 3
    num_samples = 4
    mixture_reg = 0

    pca = PCA(n_components=2)
    X_orig = X
    X_pca = pca.fit_transform(X)
    X = X_pca
    X_test = pca.transform(X_test)
    print 'explained_variance: ' + str(pca.explained_variance_ratio_.sum())

    bandwidths = np.logspace(-4,4,200)
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
        SupervisedInstanceSelectionCluster(),
        SupervisedInstanceSelection(),
    ]
    for m in methods:
        m.subset_size = num_samples
        m.num_samples = num_samples
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
                    labels[sample_idx, 0] = label_names[int(data.y[curr_idx])]
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