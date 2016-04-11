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
from methods import method
from sets import Set

class ActiveMethod(method.Saveable):
    def __init__(self,configs=MethodConfigs()):
        super(ActiveMethod, self).__init__(configs)
        self.base_learner = method.SKLRidgeRegression(configs)

    def train_and_test(self, data):
        num_items_per_iteration = self.configs.active_items_per_iteration
        active_iterations = self.configs.active_iterations
        curr_data = deepcopy(data)
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
                    all_inds = helper_functions.flatten_list_of_lists(I)
                    assert curr_data.is_train[all_inds].all()
                except AssertionError as error:
                    assert False, 'Pairwise labeling of test data isn''t implemented yet!'
                except:
                    assert not curr_data.is_labeled[I].any()
                curr_data.reveal_labels(I)
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

class RelativeActiveMethod(ActiveMethod):
    def __init__(self,configs=MethodConfigs()):
        super(RelativeActiveMethod, self).__init__(configs)

    def create_sampling_distribution(self, base_learner, data, fold_results):
        if not hasattr(data, 'pairwise_relationships'):
            data.pairwise_relationships = Set()
        I = data.is_train.nonzero()[0]
        all_pairs = Set()
        for x1 in I:
            for x2 in I:
                if x1 <= x2 or (x1,x2) in data.pairwise_relationships or (x2,x1) in data.pairwise_relationships:
                    continue
                all_pairs.add((x1,x2))
        all_pairs = np.asarray(list(all_pairs))
        d = np.zeros(all_pairs.shape[0])
        d[:] = 1
        d = d / d.sum()
        return d, all_pairs

    @property
    def prefix(self):
        return 'RelActiveRandom+' + self.base_learner.prefix









































