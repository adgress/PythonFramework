import abc
from saveable.saveable import Saveable
from configs.base_configs import MethodConfigs
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn import neighbors
from sklearn import dummy
from sklearn import grid_search
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
from timer.timer import tic,toc
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
from utility import array_functions

num_instances_for_pairs = 30

class ActiveMethod(method.Method):
    def __init__(self,configs=MethodConfigs()):
        super(ActiveMethod, self).__init__(configs)
        self.base_learner = method.SKLRidgeRegression(configs)

    def train_and_test(self, data):
        if self.configs.num_features < data.p:
            select_k_best = SelectKBest(f_regression, self.configs.num_features)
            data.x = select_k_best.fit_transform(data.x, data.true_y)
        num_items_per_iteration = self.configs.active_items_per_iteration
        active_iterations = self.configs.active_iterations
        curr_data = deepcopy(data)
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
                                constrained_methods.PairwiseConstraint(xi, xj)
                            )
                    else:
                        #all_inds = helper_functions.flatten_list_of_lists(I)
                        all_inds = I
                        assert curr_data.is_train[all_inds].all()
                        curr_data.reveal_labels(I)
                except AssertionError as error:
                    assert False, 'Pairwise labeling of test data isn''t implemented yet!'

                except Exception as err:
                    assert not curr_data.is_labeled[I].any()
            fold_results = self.base_learner.train_and_test(curr_data)
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

    @property
    def prefix(self):
        return 'ActiveRandom+' + self.base_learner.prefix

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
        #assert False, 'Use PairwiseConstraint instead of tuples'

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

    @property
    def is_pairwise(self):
        return True

    @property
    def prefix(self):
        return 'RelActiveRandom+' + self.base_learner.prefix


class RelativeActiveUncertaintyMethod(RelativeActiveMethod):
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
        all_pairs = all_pairs[inds[:min_pairs_to_keep], :]
        return all_pairs

    @property
    def prefix(self):
        return 'RelActiveUncer+' + self.base_learner.prefix

from scipy.special import expit


class OptimizationDataRelative(object):
    def __init__(self, fim_x, fim_reg, weights, deltas, reg_pairwise):
        self.fim_x = fim_x
        self.fim_reg = fim_reg
        self.weights = weights
        self.deltas = deltas
        self.reg_pairwise = reg_pairwise

def pairwise_fim(t, opt_data):
    idx = 0
    fim = np.zeros(opt_data.fim_x.shape)
    for ti, w, d in zip(t, opt_data.weights, opt_data.deltas):
        fim += ti * w * np.outer(d, d)
        idx += 1
    fim *= opt_data.reg_pairwise
    fim += opt_data.fim_x + opt_data.fim_reg
    return fim

def eval_pairwise_oed(t, opt_data):
    fim = pairwise_fim(t, opt_data)
    return np.trace(inv(fim))

def grad_pairwise_oed(t, opt_data):
    fim = pairwise_fim(t, opt_data)
    A = inv(fim)
    AA = A.dot(A)
    g = np.zeros(t.shape)
    idx = 0
    for wi, di in zip(opt_data.weights_grad, opt_data.deltas):
        g[idx] = - wi*di.T.dot(AA).dot(di)
        idx += 1
    return g

class RelativeActiveOEDMethod(RelativeActiveMethod):
    def __init__(self, configs=MethodConfigs()):
        super(RelativeActiveOEDMethod, self).__init__(configs)
        self.use_grad = False

    def create_sampling_distribution(self, base_learner, data, fold_results):
        # assert False, 'Use PairwiseConstraint instead of tuples'

        min_pairs_to_keep = 50

        I = data.is_train.nonzero()[0]
        I = np.random.choice(I, num_instances_for_pairs, False)
        #I = I[:20]
        p = data.p
        all_pairs = list()
        weights = np.zeros(100000)
        weights_grad = np.zeros(100000)
        y_pred = base_learner.predict(data).y
        diff_idx = 0
        x = self.base_learner.transform.transform(data.x)
        x_labeled = self.base_learner.transform.transform(data.x[data.is_labeled & data.is_train])
        fisher_x = x_labeled.T.dot(x_labeled)
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
                s = expit(y_pred[i] - y_pred[j])
                weights[diff_idx ] = s*(1-s)
                weights_grad[diff_idx] = 1-s
                deltas.append(x[i,:] - x[j,:])
                #fisher_pairwise += diffs[diff_idx] * np.outer(delta, delta)

        weights = weights[:diff_idx]
        weights_grad = weights_grad[:diff_idx]
        opt_data = OptimizationDataRelative(fisher_x, fisher_reg, weights, deltas, self.base_learner.C2)
        opt_data.weights_grad = weights_grad
        all_pairs = np.asarray(list(all_pairs))

        n = weights.size
        t0 = np.ones((n, 1))
        t0 /= t0.sum()
        #t0[:] = 0
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
        options = {
            'disp': True
        }
        if self.use_grad:
            results = optimize.minimize(
                lambda t: eval_pairwise_oed(t, opt_data),
                t0,
                method='SLSQP',
                jac=lambda t: grad_pairwise_oed(t, opt_data),
                options=options,
                constraints=constraints
            )
            results_eval = optimize.minimize(
                lambda t: eval_pairwise_oed(t, opt_data),
                t0,
                method='SLSQP',
                jac=None,
                options=options,
                constraints=constraints
            )
            print 'error: ' + str(array_functions.relative_error(results.x, results_eval.x))
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
        t[t < 0] = 0
        t += 1e-4
        t /= t.sum()
        return t, all_pairs

    @property
    def prefix(self):
        s = 'RelActiveOED'
        if getattr(self, 'use_grad'):
            s += '-grad'
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
        return 'RelActiveErrorMin+' + self.base_learner.prefix




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






































