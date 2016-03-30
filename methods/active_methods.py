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
from copy import deepcopy
from methods import method

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
                sampling_distribution = self.create_sampling_distribution(self.base_learner,
                                                                          curr_data,
                                                                          fold_results)
                I = array_functions.sample(sampling_distribution.size,
                                           num_items_per_iteration,
                                           sampling_distribution)
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
        return d


    def run_method(self, data):
        self.train(data)
        if data.x.shape[0] == 0:
            assert False
            self.train(data)
        return self.predict(data)

    def train(self, data):
        assert False
        pass

    def predict(self, data):
        assert False
        pass

    @property
    def prefix(self):
        return 'ActiveRandom+' + self.base_learner.prefix

