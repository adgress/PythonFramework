import abc

from pygments.lexer import include

from saveable.saveable import Saveable
from configs.base_configs import MethodConfigs
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn import neighbors
from sklearn import dummy
from sklearn import grid_search
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise
import numpy as np
from numpy.linalg import norm
from copy import deepcopy
from results_class.results import Output
from results_class.results import FoldResults
from results_class import results as results_lib
from data_sets import create_data_split
from data import data as data_lib
from utility import array_functions
from metrics import metrics
import collections
import scipy
from timer.timer import tic,toc, toc_str
from utility import helper_functions
from copy import deepcopy
import cvxpy as cvx
from methods import method
from numpy.linalg import *
from scipy import optimize
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
import constrained_methods
from constrained_methods import PairwiseConstraint
from utility import array_functions

num_instances_for_pairs = 20

class ActiveMethod(method.Method):
    def __init__(self,configs=MethodConfigs()):
        super(ActiveMethod, self).__init__(configs)
        self.base_learner = method.SKLRidgeRegression(configs)
        self.fix_model = False

    def train_and_test(self, data):
        if self.configs.num_features is not None and self.configs.num_features < data.p:
            select_k_best = SelectKBest(f_regression, self.configs.num_features)
            data.x = select_k_best.fit_transform(data.x, data.true_y)
        num_items_per_iteration = self.configs.active_items_per_iteration
        active_iterations = self.configs.active_iterations
        curr_data = deepcopy(data)
        '''
        if curr_data.p > 4:
            pca = PCA(n_components=4)
            curr_data.x = pca.fit_transform(curr_data.x)
        '''
        if hasattr(self.base_learner, 'add_random_guidance'):
            self.base_learner.add_random_pairwise = True
            self.base_learner.add_random_guidance(curr_data)
            self.base_learner.add_random_pairwise = False
        active_fold_results = results_lib.ActiveFoldResults(active_iterations)
        for iter_idx in range(active_iterations):
            I = np.empty(0)
            if iter_idx > 0:
                sampling_distribution, items = self.create_sampling_distribution(self.base_learner,
                                                                                 curr_data,
                                                                                 fold_results)
                I = array_functions.sample(items,
                                           num_items_per_iteration,
                                           sampling_distribution)
                try:
                    if self.is_pairwise:
                        if not hasattr(data, 'pairwise_relationships'):
                            data.pairwise_relationships = []

                        for i, j in I:
                            xi = data.x[i,:]
                            xj = data.x[j,:]
                            curr_data.pairwise_relationships = np.append(
                                curr_data.pairwise_relationships,
                                constrained_methods.PairwiseConstraint(xi, xj, i, j)
                            )
                    else:
                        #all_inds = helper_functions.flatten_list_of_lists(I)
                        all_inds = I
                        assert curr_data.is_train[all_inds].all()
                        curr_data.reveal_labels(I)
                except AssertionError as error:
                    assert False, 'Pairwise labeling of test data isn''t implemented yet!'

                except Exception as err:
                    assert False, 'Other Error'
                    assert not curr_data.is_labeled[I].any()
            if iter_idx == 0 or not self.fix_model:
                fold_results = self.base_learner.train_and_test(curr_data)
            else:
                fold_results = self.base_learner.run_method(curr_data)
            active_iteration_results = results_lib.ActiveIterationResults(fold_results,I)
            active_fold_results.set(active_iteration_results, iter_idx)
        return active_fold_results

    def create_sampling_distribution(self, base_learner, data, fold_results):
        I = data.is_train & ~data.is_labeled
        d = np.zeros(data.y.shape)
        d[I] = 1
        d = d / d.sum()
        return d, d.size

    @property
    def is_pairwise(self):
        return False

    def run_method(self, data):
        assert False, 'Not implemented for ActiveMethod'

    def train(self, data):
        assert False, 'Not implemented for ActiveMethod'
        pass

    def predict(self, data):
        assert False, 'Not implemented for ActiveMethod'
        pass

    def active_options_suffix(self):
        s = ''
        s += '_n=' + str(getattr(self.configs, 'num_starting_labels', '???'))
        s += '_items=' + str(self.configs.active_items_per_iteration)
        s += '_iters=' + str(self.configs.active_iterations)
        return s

    @property
    def prefix(self):
        s = 'ActiveRandom'
        if getattr(self, 'fix_model', False):
            s += '_fixed-model'
        s += self.active_options_suffix()
        s += '+' + self.base_learner.prefix
        return s


class ClusterActiveMethod(ActiveMethod):
    def __init__(self, configs=MethodConfigs()):
        super(ClusterActiveMethod, self).__init__(configs)
        self.transform = StandardScaler()
        self.use_labeled = True
        self.cluster_scale = 10

    def create_clustering(self, X, n_items):
        k_means = KMeans(n_clusters=n_items)
        X_cluster_space = k_means.fit_transform(X)
        cluster_inds = k_means.fit_predict(X)
        return X_cluster_space, cluster_inds

    def get_cluster_centroids(self, X):
        p = X.shape[1]
        idx = np.zeros(p)
        for i in range(p):
            xi = X[:, i]
            idx[i] = np.argmin(xi)
        return idx.astype(np.int)


    def create_sampling_distribution(self, base_learner, data, fold_results):
        cluster_scale = self.cluster_scale
        n_items = self.configs.active_items_per_iteration
        I = data.is_train & ~data.is_labeled
        if self.configs.target_labels is not None:
            I &= data.get_transfer_inds(self.configs.target_labels)
        I = I.nonzero()[0]
        if I.size > 1000:
            I = np.random.choice(I, int(I.size*.5), replace=False)
            print 'subsampling target data: ' + str(I.size)
        X_sub = data.x[I, :]
        X_cluster_space, _ = self.create_clustering(X_sub, int(cluster_scale*n_items))
        d = np.zeros(data.y.shape)
        centroid_idx = self.get_cluster_centroids(X_cluster_space)
        to_use = centroid_idx[:n_items]
        d[I[to_use]] = 1
        d = d / d.sum()
        return d, d.size

    @property
    def prefix(self):
        s = 'ActiveCluster'
        s += self.active_options_suffix()
        s += '_scale=' + str(self.cluster_scale)
        s += '+' + self.base_learner.prefix
        return s


from instance_selection import SupervisedInstanceSelectionClusterGraph
class ClusterPurityActiveMethod(ClusterActiveMethod):
    def __init__(self, configs=MethodConfigs()):
        super(ClusterActiveMethod, self).__init__(configs)
        self.transform = StandardScaler()
        self.use_target_variance = True
        self.use_density = False
        self.use_instance_selection = True
        self.instance_selector = SupervisedInstanceSelectionClusterGraph(deepcopy(configs))
        self.use_warm_start = False
        self.use_oracle_labels = True

    def get_cluster_purity(self, cluster_ids, y, classification=False):
        num_clusters = cluster_ids.max()+1
        v = np.zeros(num_clusters)
        m = np.zeros(num_clusters)
        for i in range(num_clusters):
            yi = y[cluster_ids == i]
            if classification:
                yi = array_functions.to_binary_vec(yi)
            v[i] = np.std(yi)
            m[i] = yi.size
        return v, m

    def estimate_variance(self, learner, data):
        num_splits = 30
        predictions = np.zeros((data.n, num_splits))
        for i in range(num_splits):
            sub_data = data.rand_sample(.1)
            learner.train(sub_data)
            predictions[:, i] = learner.predict(data).y
        vars = np.std(predictions, 1)
        return vars

    def estimate_density(self, data):
        from methods import density
        bandwidths = np.logspace(-2, 2)
        best_bandwidth, vals = density.get_best_bandwidth(data.x, bandwidths)
        d = density.compute_density(data.x, None, best_bandwidth)
        return d

    def create_sampling_distribution(self, base_learner, data, fold_results):
        cluster_scale = self.cluster_scale
        source_learner = deepcopy(self.base_learner)
        source_data = data.get_transfer_subset(self.configs.source_labels)
        if source_data.n > 1000:
            source_data = source_data.rand_sample(.2)
            print 'subsampling source data: ' + str(source_data.n)
        if source_data.is_regression:
            source_data.data_set_ids[:] = self.configs.target_labels[0]
        else:
            source_data.change_labels(self.configs.source_labels, self.configs.target_labels)
        tic()
        source_learner.train_and_test(source_data)
        print 'train source time: ' + toc_str()
        target_data = data.get_transfer_subset(self.configs.target_labels, include_unlabeled=True)
        y_pred = source_learner.predict(data).y
        if self.use_oracle_labels:
            y_pred = data.true_y.copy()

        n_items = self.configs.active_items_per_iteration
        I = data.is_train
        if not self.use_warm_start:
            I &= ~data.is_labeled
        if self.configs.target_labels is not None:
            I &= data.get_transfer_inds(self.configs.target_labels)
        I = I.nonzero()[0]
        if I.size > 1000:
            I = np.random.choice(I, int(I.size*.5), replace=False)
            print 'subsampling target data: ' + str(I.size)

        labeled_target_data = deepcopy(data.get_subset(I))
        instances_to_keep = labeled_target_data.is_labeled
        labeled_target_data.set_train()
        labeled_target_data.is_noisy = array_functions.false(labeled_target_data.n)

        labeled_target_data.y = y_pred[I].copy()
        labeled_target_data.true_y = y_pred[I].copy()
        labeled_target_data.y_orig = y_pred[I].copy()
        labeled_target_data.instances_to_keep = instances_to_keep

        #labeled_target_data.y_orig = labeled_target_data.true_y.copy()
        if self.use_instance_selection:
            self.instance_selector.subset_size = n_items
            self.instance_selector.num_samples = n_items
            self.instance_selector.configs.use_validation = False
            self.instance_selector.configs.use_training = True
            self.instance_selector.train_and_test(labeled_target_data)
            is_selected = self.instance_selector.predict(labeled_target_data).is_selected
            scores = np.ones(is_selected.size)
            #Lower score is better
            scores[is_selected] = 0
            scores_sorted_inds = np.argsort(scores)
            print ''
        elif self.use_density:
            target_learner = deepcopy(self.base_learner)
            target_learner.train_and_test(labeled_target_data)
            vars = self.estimate_variance(target_learner, labeled_target_data, )
            densities = self.estimate_density(labeled_target_data)
        else:
            X_sub = data.x[I, :]
            tic()
            X_cluster_space, cluster_ids = self.create_clustering(
                X_sub,
                int(cluster_scale * self.configs.active_items_per_iteration)
            )
            print 'cluster target time: ' + toc_str()
            vars, cluster_n = self.get_cluster_purity(cluster_ids, y_pred[I], not target_data.is_regression)
            true_vars, true_cluster_n = self.get_cluster_purity(cluster_ids, data.true_y[I], not target_data.is_regression)
            if self.use_target_variance:
                vars = true_vars
            centroid_idx = self.get_cluster_centroids(X_cluster_space)
            densities = cluster_n
        if self.use_instance_selection:
            pass
        else:
            scores = vars / densities
            scores_sorted_inds = np.argsort(scores)

        # Don't sample instances if cluster size is 1
        if not self.use_density and not self.use_instance_selection:
            scores[cluster_n <= .005*I.size] = np.inf
            to_use = centroid_idx[scores_sorted_inds[:n_items]]
        else:
            to_use = scores_sorted_inds[:n_items]

        d = np.zeros(data.y.shape)
        d[I[to_use]] = 1
        d = d / d.sum()
        return d, d.size

    @property
    def prefix(self):
        s = 'ActiveClusterPurity'
        use_inst_sel = getattr(self, 'use_instance_selection', False)
        if use_inst_sel:
            s += '-instanceSel'
        else:
            if getattr(self, 'use_target_variance', False):
                s += '-targetVar'
            if getattr(self, 'use_density', False):
                s += '-density'
        if getattr(self, 'use_warm_start', False):
            s += '_warmStart'
        s += self.active_options_suffix()
        if not use_inst_sel:
            s += '_scale=' + str(self.cluster_scale)
        if getattr(self, 'use_oracle_labels'):
            s += '_oracleY'
        s += '+' + self.base_learner.prefix
        return s


class OptimizationData(object):
    def __init__(self, x, C):
        self.x = x
        self.C = C
        self.x_labeled = None


def eval_oed(t, opt_data):
    x = opt_data.x
    n, p = x.shape
    M = opt_data.C * np.eye(p)
    if opt_data.x_labeled is not None:
        xl = opt_data.x_labeled
        M += xl.T.dot(xl)
    for i in range(n):
        M += t[i]*np.outer(x[i,:], x[i,:])

    return np.trace(inv(M))

class OEDLinearActiveMethod(ActiveMethod):
    def __init__(self, configs=MethodConfigs()):
        super(OEDLinearActiveMethod, self).__init__(configs)
        self.transform = StandardScaler()
        self.use_labeled = True

    def create_sampling_distribution(self, base_learner, data, fold_results):
        is_train_unlabeled = data.is_train & (~data.is_labeled)
        is_train_labeled = data.is_train & data.is_labeled
        inds = np.nonzero(is_train_unlabeled)[0]
        inds = inds[:50]
        I = array_functions.false(data.n)
        I[inds] = True
        x = data.x[I, :]
        x_labeled = data.x[is_train_labeled, :]
        if self.use_labeled:
            x_all = np.vstack((x, x_labeled))
            self.transform.fit(x_all)
            x = self.transform.transform(x)
            x_labeled = self.transform.transform(x_labeled)
        else:
            x = self.transform.fit_transform(x)
        C = base_learner.params['alpha']
        n = I.sum()
        t0 = np.zeros((n,1))
        opt_data = OptimizationData(x, C)
        if self.use_labeled:
            opt_data.x_labeled = x_labeled
        constraints = [
            {
                'type': 'eq',
                'fun': lambda t: t.sum() - 1
            },
            {
                'type': 'ineq',
                'fun': lambda t: t
            }
        ]
        options = {}
        results = optimize.minimize(
            lambda t: eval_oed(t, opt_data),
            t0,
            method='SLSQP',
            jac=None,
            options=options,
            constraints=constraints
        )
        if results.success:
            t = results.x
        else:
            print 'OED Optimization failed'
            t = np.ones(n)
        t[t < 0] = 0
        t += 1e-4
        t /= t.sum()
        return t, inds

    @property
    def prefix(self):
        s = 'OED+' + self.base_learner.prefix
        if self.use_labeled:
            s += '_use-labeled'
        if getattr(self, 'fix_model', False):
            s += '_fixed-model'
        return s

class RelativeActiveMethod(ActiveMethod):
    def __init__(self,configs=MethodConfigs()):
        super(RelativeActiveMethod, self).__init__(configs)

    def create_sampling_distribution(self, base_learner, data, fold_results):
        all_pairs = self.create_pairs(data, base_learner)
        d = np.zeros(all_pairs.shape[0])
        d[:] = 1
        d = d / d.sum()
        return d, all_pairs

    def create_pairs(self, data, base_learner):
        #assert False, 'Use PairwiseRe
        I = data.is_train.nonzero()[0]
        I = np.random.choice(I, num_instances_for_pairs, False)
        all_pairs = set()
        for i in I:
            for j in I:
                if data.true_y[i] >= data.true_y[j]:
                    continue
                #TODO: Don't add redundant pairs
                all_pairs.add((i, j))
        all_pairs = np.asarray(list(all_pairs))
        return all_pairs

    def active_options_suffix(self):
        n = str(self.num_labels[0])
        iterations = str(self.configs.active_iterations)
        items_per_iteration = str(self.configs.active_items_per_iteration)
        s = n + '-' + iterations + '-' + items_per_iteration
        return s

    @property
    def is_pairwise(self):
        return True

    @property
    def prefix(self):
        s = 'RelActiveRandom'
        if getattr(self, 'fix_model', False):
            s += '_fixed-model'
        s += '-' + self.active_options_suffix()
        s += '+' + self.base_learner.prefix
        return s


class RelativeActiveUncertaintyMethod(RelativeActiveMethod):
    def __init__(self, configs=MethodConfigs()):
        super(RelativeActiveUncertaintyMethod, self).__init__(configs)
        self.use_oracle = False
        self.use_largest_delta = False

    def create_pairs(self, data, base_learner):
        # assert False, 'Use Pairwisel
        min_pairs_to_keep = 50

        I = data.is_train.nonzero()[0]
        I = np.random.choice(I, num_instances_for_pairs, False)
        #I = I[:num_instances_for_pairs]
        all_pairs = list()
        diffs = np.zeros(100000)
        if self.use_oracle:
            y_pred = data.true_y
        else:
            y_pred = base_learner.predict(data).y
        diff_idx = 0
        for i in I:
            for j in I:
                if data.true_y[i] >= data.true_y[j]:
                    continue
                # TODO: Don't add redundant pairs
                diff_idx += 1
                all_pairs.append((i, j))
                diffs[diff_idx] = np.abs(y_pred[i] - y_pred[j])
        diffs = diffs[0:diff_idx]
        inds = np.argsort(diffs)
        if self.use_largest_delta:
            inds = inds[::-1]
        all_pairs = np.asarray(list(all_pairs))
        all_pairs = all_pairs[inds[:self.configs.active_items_per_iteration], :]
        return all_pairs

    @property
    def prefix(self):
        s = 'RelActiveUncer'
        if getattr(self, 'use_largest_delta', False):
            s += '-largest'
        if getattr(self, 'use_oracle', False):
            s += '-oracle'
        if getattr(self, 'fix_model', False):
            s += '_fixed-model'
        s += '-' + self.active_options_suffix()
        s += '+' + self.base_learner.prefix
        return s

from scipy.special import expit


class OptimizationDataRelative(object):
    def __init__(self, fim_x, fim_reg, weights, deltas, reg_pairwise):
        self.fim_x = fim_x
        self.fim_reg = fim_reg
        self.weights = weights
        self.deltas = deltas
        self.reg_pairwise = float(reg_pairwise)

def pairwise_fim(t, opt_data):
    idx = 0
    fim = np.zeros(opt_data.fim_x.shape)
    for ti, w, d in zip(t, opt_data.weights, opt_data.deltas):
        fim += ti * w * np.outer(d, d)
        idx += 1
    n = float(opt_data.num_items)
    if opt_data.weights_labeled is not None:
        for w, d in zip(opt_data.weights_labeled, opt_data.deltas_labeled):
            fim += w * np.outer(d,d)
        n += len(opt_data.weights_labeled)
    #n = 1
    fim *= (opt_data.reg_pairwise/n)
    fim += opt_data.fim_x + opt_data.fim_reg
    return fim

def eval_pairwise_oed(t, opt_data):
    fim = pairwise_fim(t, opt_data)
    n = opt_data.num_items
    if opt_data.weights_labeled is not None:
        n += len(opt_data.weights_labeled)
    try:
        inv_fim = inv(fim)
    except Exception as e:
        inv_fim = inv(fim + 1e-4 * np.eye(fim.shape[0]))
    if opt_data.oed_method == 'E':
        vals, vecs = eigh(inv_fim)
        v = vals.max()
    elif False:
        v = np.linalg.det(inv_fim)
    else:
        v = np.trace(inv_fim)
    return v
    #return np.trace(fim)

def grad_pairwise_oed(t, opt_data):
    fim = pairwise_fim(t, opt_data)
    n = opt_data.num_items
    if opt_data.weights_labeled is not None:
        n += len(opt_data.weights_labeled)
    if opt_data.oed_method == 'E':
        assert False
    else:
        try:
            A = inv(fim)
        except:
            A = inv(fim + 1e-4 * np.eye(fim.shape[0]))

        AA = A.dot(A)
        g = np.zeros(t.shape)
        idx = 0
        for wi, di in zip(opt_data.weights, opt_data.deltas):
            g[idx] = - wi*di.T.dot(AA).dot(di)
            #g[idx] = wi*di.T.dot(di)
            idx += 1
        C = (opt_data.reg_pairwise/n)
        g *= C
        return g

class RelativeActiveOEDMethod(RelativeActiveMethod):
    def __init__(self, configs=MethodConfigs()):
        super(RelativeActiveOEDMethod, self).__init__(configs)
        self.use_grad = True
        #self.oed_method = 'E'
        self.oed_method = None
        self.use_labeled = True
        self.use_true_y = getattr(configs, 'use_true_y', False)

    def create_sampling_distribution(self, base_learner, data, fold_results):
        # assert False, 'Use PairwiseConstraint instead of tuples'

        min_pairs_to_keep = 50

        I = data.is_train.nonzero()[0]
        I = np.random.choice(I, num_instances_for_pairs, False)
        #I = I[:num_instances_for_pairs]
        #I = I[:20]
        p = data.p
        all_pairs = list()
        weights = np.zeros(100000)
        y_pred = base_learner.predict(data).y
        diff_idx = 0
        x = self.base_learner.transform.transform(data.x)
        x_labeled = self.base_learner.transform.transform(data.x[data.is_labeled & data.is_train])
        fisher_x = x_labeled.T.dot(x_labeled) / x_labeled.shape[0]
        fisher_reg = self.base_learner.C*np.eye(p)
        deltas = list()
        #fisher_pairwise = np.zeros((p,p))
        for i in I:
            for j in I:
                if data.true_y[i] >= data.true_y[j]:
                    continue
                # TODO: Don't add redundant pairs
                diff_idx += 1
                all_pairs.append((i, j))
                #weights[diff_idx] = expit(y_pred[i] - y_pred[j])
                diff = y_pred[i] - y_pred[j]
                if self.use_true_y:
                    diff = data.true_y[i] - data.true_y[j]
                s = expit(diff)
                weights[diff_idx ] = s*(1-s)
                deltas.append(x[i,:] - x[j,:])
                #fisher_pairwise += diffs[diff_idx] * np.outer(delta, delta)
        weights_labeled = None
        deltas_labeled = None
        if self.use_labeled and len(data.pairwise_relationships) > 0:
            transform = self.base_learner.transform
            x_low, x_high, _, _ = PairwiseConstraint.generate_pairs_for_scipy_optimize(data.pairwise_relationships)
            y_low = self.base_learner.predict_x(x_low).y
            y_high = self.base_learner.predict_x(x_high).y
            deltas_labeled = transform.transform(x_low) - transform.transform(x_high)
            s_labeled = expit(y_low - y_high)
            weights_labeled = s_labeled*(1-s_labeled)
        weights = weights[:diff_idx]
        opt_data = OptimizationDataRelative(fisher_x, fisher_reg, weights, deltas, self.base_learner.C2)
        opt_data.weights_labeled = weights_labeled
        opt_data.deltas_labeled = deltas_labeled
        opt_data.oed_method = self.oed_method
        opt_data.num_items = self.configs.active_items_per_iteration
        if data.pairwise_relationships.size == 0:
            opt_data.reg_pairwise = 1
        all_pairs = np.asarray(list(all_pairs))

        n = weights.size
        t0 = np.ones(n)
        t0 /= t0.sum()
        C = self.configs.active_items_per_iteration
        #C = .01
        t0 *= C
        #t0[:] = 0
        constraints = [
            {
                'type': 'eq',
                'fun': lambda t: t.sum() - C
            },
            {
                'type': 'ineq',
                'fun': lambda t: t
            }
        ]
        options = {
            'disp': False,
            'maxiter': 1000
        }
        if self.use_grad and False:
            results = optimize.minimize(
                lambda t: eval_pairwise_oed(t, opt_data),
                t0,
                method='SLSQP',
                #jac=lambda t: grad_pairwise_oed(t, opt_data),
                jac = None,
                options=options,
                constraints=constraints
            )
            grad_err = scipy.optimize.check_grad(
                lambda t: eval_pairwise_oed(t, opt_data),
                lambda t: grad_pairwise_oed(t, opt_data),
                results.x
            )
            g = grad_pairwise_oed(t0, opt_data)
            g_approx = optimize.approx_fprime(t0, lambda t: eval_pairwise_oed(t, opt_data), 1e-6)
            print 'RelativeOED grad err: ' + str(grad_err)
            '''
            if not results.success:
                print 'Gradient failed - using eval'
                results_eval = optimize.minimize(
                    lambda t: eval_pairwise_oed(t, opt_data),
                    t0,
                    method='SLSQP',
                    jac=None,
                    options=options,
                    constraints=constraints
                )
                results = results_eval
                if results.success:
                    print 'eval success'
                else:
                    print 'eval failed'
            '''
            '''
            t_vals = [results.x, results_eval.x]
            for v in t_vals:
                v[v < 0] = 0
                v /= v.sum()
            v0 = t_vals[0]
            v1 = t_vals[1]
            print 'error: ' + str(array_functions.relative_error(v[0], v[1]))
            '''
        else:
            results = optimize.minimize(
                lambda t: eval_pairwise_oed(t, opt_data),
                t0,
                method='SLSQP',
                jac=None,
                options=options,
                constraints=constraints
            )

        if results.success:
            t = results.x
            #print 'error: ' + str(array_functions.relative_error(results.x, results_grad.x))
        else:
            print 'OED Optimization failed'
            t = np.ones(n)
        #print t.sum()
        t_old = t
        t[t < 0] = 0
        t += np.random.uniform(1e-7,1e-6, t.size)
        t /= t.sum()
        #print np.sort(t)[-40:]
        best_inds = np.argsort(t)[-self.configs.active_items_per_iteration:]
        t = t[best_inds]
        all_pairs =all_pairs[best_inds]
        t /= t.sum()
        print t
        return t, all_pairs

    @property
    def prefix(self):
        s = 'RelActiveOED'
        if getattr(self, 'oed_method', None) is not None:
            s += '-' + self.oed_method
        if getattr(self, 'use_grad'):
            s += '-grad'
        if getattr(self, 'use_labeled'):
            s += '-labeled'
        if getattr(self, 'use_true_y'):
            s += '-trueY'
        if getattr(self, 'fix_model', False):
            s += '_fixed-model'
        s += '-' + self.active_options_suffix()
        s += '+' + self.base_learner.prefix
        return s

class RelativeActiveErrorMinMethod(RelativeActiveMethod):
    def create_pairs(self, data, base_learner):
        # assert False, 'Use PairwiseConstraint instead of tuples'

        min_pairs_to_keep = 50

        I = data.is_train.nonzero()[0]
        I = np.random.choice(I, num_instances_for_pairs, False)
        all_pairs = list()
        diffs = np.zeros(100000)
        y_pred = base_learner.predict(data).y
        diff_idx = 0
        for i in I:
            for j in I:
                if data.true_y[i] >= data.true_y[j]:
                    continue
                # TODO: Don't add redundant pairs
                diff_idx += 1
                all_pairs.append((i, j))
                diffs[diff_idx] = np.abs(y_pred[i] - y_pred[j])
        diffs = diffs[0:diff_idx]
        inds = np.argsort(diffs)
        all_pairs = np.asarray(list(all_pairs))
        all_pairs = all_pairs[inds[:50], :]
        return all_pairs

    @property
    def prefix(self):
        s = 'RelActiveErrorMin'
        if getattr(self, 'fix_model', False):
            s += '_fixed-model'
        s += '-' + self.active_options_suffix()
        s += '+' +self.base_learner.prefix
        return s




class IGRelativeActiveMethod(RelativeActiveMethod):
    def __init__(self,configs=MethodConfigs()):
        super(IGRelativeActiveMethod, self).__init__(configs)

    def create_sampling_distribution(self, base_learner, data, fold_results):
        all_pairs = self.create_pairs(data)
        d = np.zeros(all_pairs.shape[0])
        for x1, x2 in all_pairs:
            pass
        d = d / d.sum()
        return d, all_pairs

    @property
    def prefix(self):
        return 'RelActiveRandom+' + self.base_learner.prefix

