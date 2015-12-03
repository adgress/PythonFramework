__author__ = 'Aubrey'

import abc
from saveable.saveable import Saveable
from configs.base_configs import MethodConfigs
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn import dummy
from sklearn import grid_search
import numpy as np
from copy import deepcopy
from results_class.results import Output
from results_class.results import FoldResults
from results_class import results as results_lib
from data_sets import create_data_split
from data import data as data_lib
import collections


class Method(Saveable):

    def __init__(self,configs=MethodConfigs()):
        super(Method, self).__init__(configs)
        self.configs = configs
        self._params = []
        self.cv_params = {}
        self.is_classifier = True
        self.experiment_results_class = results_lib.ExperimentResults
        self.cv_use_data_type = True

    @property
    def params(self):
        return self._params

    def set_params(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def run_method(self, data):
        self.train(data)
        return self.predict(data)

    def _create_cv_splits(self,data):
        data_splitter = create_data_split.DataSplitter()
        num_splits = 5
        perc_train = .8
        is_regression = data.is_regression
        if self.cv_use_data_type:
            splits = data_splitter.generate_splits(data.y,num_splits,perc_train,is_regression,data.is_target)
        else:
            splits = data_splitter.generate_splits(data.y,num_splits,perc_train,is_regression)
        return splits

    def run_cross_validation(self,data):
        train_data = data.get_subset(data.is_train & data.is_labeled)

        splits = self._create_cv_splits(train_data)
        data_and_splits = data_lib.SplitData(train_data,splits)
        param_grid = list(grid_search.ParameterGrid(self.cv_params))
        if not self.cv_params:
            return param_grid[0]
        param_results = []
        for i in range(len(param_grid)):
            param_results.append(self.experiment_results_class())
        for i in range(len(splits)):
            curr_split = data_and_splits.get_split(i)
            for param_idx, params in enumerate(param_grid):
                self.set_params(**params)
                results = self.run_method(curr_split)
                fold_results = FoldResults()
                fold_results.prediction = results
                param_results[param_idx].append(fold_results)

        errors = np.empty(len(param_grid))
        for i in range(len(param_grid)):
            agg_results = param_results[i].aggregate_error(self.configs.loss_function)
            errors[i] = agg_results.mean

        min_error = errors.min()
        best_params = param_grid[errors.argmin()]
        return best_params



    def train_and_test(self, data):
        best_params = self.run_cross_validation(data)
        self.set_params(**best_params)
        output = self.run_method(data)
        f = FoldResults()
        f.prediction = output
        return f


    @abc.abstractmethod
    def train(self, data):
        pass

    @abc.abstractmethod
    def predict(self, data):
        pass

class ScikitLearnMethod(Method):

    _short_name_dict = {
        'Ridge': 'RidgeReg',
        'DummyClassifier': 'DumClass',
        'LogisticRegression': 'LogReg'
    }

    def __init__(self,configs=MethodConfigs(),skl_method=None):
        super(ScikitLearnMethod, self).__init__(configs)
        self.skl_method = skl_method

    def train(self, data):
        labeled_train = data.labeled_training_data()
        self.skl_method.fit(labeled_train.x, labeled_train.y)

    def predict(self, data):
        o = Output(data)
        o.y = self.skl_method.predict(data.x)
        return o

    def set_params(self, **kwargs):
        super(ScikitLearnMethod,self).set_params(**kwargs)
        self.skl_method.set_params(**kwargs)

    def _skl_method_name(self):
        return repr(self.skl_method).split('(')[0]

    @property
    def prefix(self):
        return "SKL-" + ScikitLearnMethod._short_name_dict[self._skl_method_name()]

class SKLRidgeRegression(ScikitLearnMethod):
    def __init__(self,configs=None):
        super(SKLRidgeRegression, self).__init__(configs, linear_model.Ridge())
        self.cv_params['alpha'] = 10**np.asarray(range(-8,8),dtype='float64')
        self.set_params(alpha=0,fit_intercept=True,normalize=True)

class SKLLogisticRegression(ScikitLearnMethod):
    def __init__(self,configs=None):
        super(SKLLogisticRegression, self).__init__(configs, linear_model.LogisticRegression())
        self.cv_params['C'] = 10**np.asarray(list(reversed(range(-5, 5))),dtype='float64')
        self.set_params(C=0,fit_intercept=True,penalty='l2')

class SKLGuessClassifier(ScikitLearnMethod):
    def __init__(self,configs=None):
        super(dummy.DummyClassifier('uniform'), self).__init__(configs)

class SKLMeanRegressor(ScikitLearnMethod):
    def __init__(self,configs=None):
        super(dummy.DummyRegressor('mean'), self).__init__(configs)