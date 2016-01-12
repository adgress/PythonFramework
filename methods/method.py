__author__ = 'Aubrey'

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


#from pyqt_fit import nonparam_regression
#from pyqt_fit import npr_methods

class Method(Saveable):

    def __init__(self,configs=MethodConfigs()):
        super(Method, self).__init__(configs)
        self._params = []
        self.cv_params = {}
        self.is_classifier = True
        self.experiment_results_class = results_lib.ExperimentResults
        self.cv_use_data_type = True
        self._estimated_error = None
        self.quiet = True
        self.best_params = None

    @property
    def params(self):
        return self._params

    @property
    def estimated_error(self):
        return self._estimated_error

    @estimated_error.setter
    def estimated_error(self, value):
        self._estimated_error = value

    def set_params(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self._params = kwargs

    def run_method(self, data):
        self.train(data)
        if data.x.shape[0] == 0:
            assert False
            self.train(data)
        return self.predict(data)

    def _create_cv_splits(self,data):
        data_splitter = create_data_split.DataSplitter()
        num_splits = 10
        perc_train = .8
        is_regression = data.is_regression
        if self.cv_use_data_type:
            splits = data_splitter.generate_splits(data.y,num_splits,perc_train,is_regression,data.is_target)
        else:
            splits = data_splitter.generate_splits(data.y,num_splits,perc_train,is_regression)
        return splits

    def run_cross_validation(self,data):
        train_data = data.get_subset(data.is_train)

        splits = self._create_cv_splits(train_data)
        data_and_splits = data_lib.SplitData(train_data,splits)
        param_grid = list(grid_search.ParameterGrid(self.cv_params))
        if not self.cv_params:
            return param_grid[0], None
        param_results =[self.experiment_results_class(len(splits)) for i in range(len(param_grid))]
        unused_breakpoint = 1
        for i in range(len(splits)):
            curr_split = data_and_splits.get_split(i)
            curr_split.remove_test_labels()
            for param_idx, params in enumerate(param_grid):
                self.set_params(**params)
                results = self.run_method(curr_split)
                fold_results = FoldResults()
                fold_results.prediction = results
                param_results[param_idx].set(fold_results, i)
                #Make sure error can be computed
                #param_results[param_idx].aggregate_error(self.configs.cv_loss_function)

        errors = np.empty(len(param_grid))
        for i in range(len(param_grid)):
            agg_results = param_results[i].aggregate_error(self.configs.cv_loss_function)
            errors[i] = agg_results.mean

        min_error = errors.min()
        best_params = param_grid[errors.argmin()]
        if not self.quiet:
            print best_params
        self.best_params = best_params
        return [best_params, min_error]

    def process_data(self, data):
        labels_to_keep = np.empty(0)
        t = getattr(self.configs,'target_labels',None)
        s = getattr(self.configs,'source_labels',None)
        if t is not None and t.size > 0:
            labels_to_keep = np.concatenate((labels_to_keep,t))
        if s is not None and s.size > 0:
            s = s.ravel()
            labels_to_keep = np.concatenate((labels_to_keep,s))
            inds = array_functions.find_set(data.y,s)
            data.type[inds] = data_lib.TYPE_SOURCE
            data.is_train[inds] = True
        if labels_to_keep.size > 0:
            data = data.get_transfer_subset(labels_to_keep,include_unlabeled=True)
        return data


    def train_and_test(self, data):
        self.should_plot_g = False
        data = self.process_data(data)
        best_params, min_error = self.run_cross_validation(data)
        self.set_params(**best_params)
        self.should_plot_g = True
        output = self.run_method(data)
        f = FoldResults()
        f.prediction = output
        f.estimated_error = min_error
        self.estimated_error = min_error
        for key,value in best_params.iteritems():
            setattr(f,key,value)
        return f


    @abc.abstractmethod
    def train(self, data):
        pass

    @abc.abstractmethod
    def predict(self, data):
        pass

    def predict_loo(self, data):
        assert False, 'Not implemented!'

class ModelSelectionMethod(Method):
    def __init__(self, configs=None):
        super(ModelSelectionMethod, self).__init__(configs)
        self.methods = []
        self.chosen_method_idx = None

    @property
    def selected_method(self):
        assert self.chosen_method_idx is not None
        return self.methods[self.chosen_method_idx]

    def train(self, data):
        assert len(self.methods) > 0
        estimated_errors = np.zeros(len(self.methods))
        for i, method in enumerate(self.methods):
            results = method.train_and_test(data)
            estimated_errors[i] = method.estimated_error
        self.chosen_method_idx = estimated_errors.argmin()
        print 'Chose: ' + str(self.selected_method.__class__)


    def predict(self, data):
        return self.selected_method.predict(data)

    @property
    def prefix(self):
        return 'ModelSelection'

class NadarayaWatsonMethod(Method):
    def __init__(self,configs=MethodConfigs()):
        super(NadarayaWatsonMethod, self).__init__(configs)
        self.cv_params['sigma'] = 10**np.asarray(range(-8,8),dtype='float64')
        #self.sigma = 1
        self.metric = 'euclidean'
        if 'metric' in configs.__dict__:
            self.metric = configs.metric
        self.instance_weights = None
        #self.metric = 'cosine'

    def compute_kernel(self,x,y):
        #TODO: Optimize this for cosine similarity using cross product and matrix multiplication
        W = pairwise.pairwise_distances(x,y,self.metric)
        W = np.square(W)
        W = -self.sigma * W
        W = np.exp(W)
        return W
        #return pairwise.rbf_kernel(x,y,self.sigma)

    def train(self, data):
        is_labeled_train = data.is_train & data.is_labeled
        labeled_train = data.labeled_training_data()
        x_labeled = labeled_train.x
        self.x = x_labeled
        self.y = labeled_train.y
        self.is_classifier = not data.is_regression

        if 'instance_weights' in data.__dict__ and data.instance_weights is not None:
            self.instance_weights = data.instance_weights[is_labeled_train]

    def predict(self, data):
        o = Output(data)
        #W = pairwise.rbf_kernel(data.x,self.x,self.sigma)
        W = self.compute_kernel(data.x, self.x)
        if self.instance_weights is not None:
            W = W*self.instance_weights
        '''
        W = array_functions.replace_invalid(W,0,0)
        D = W.sum(1)
        D[D==0] = 1
        D_inv = 1 / D
        array_functions.replace_invalid(D_inv,x_min=1,x_max=1)
        S = (W.swapaxes(0, 1) * D_inv).swapaxes(0, 1)
        '''
        S = array_functions.make_smoothing_matrix(W)
        if not data.is_regression:
            fu = np.zeros((data.n,self.y.max()+1))
            for i in np.unique(self.y):
                I = self.y == i
                Si = S[:,I]
                fu_i = Si.sum(1)
                fu[:,i] = fu_i
            fu2 = fu
            fu = array_functions.replace_invalid(fu,0,1)
            fu = array_functions.normalize_rows(fu)
            o.fu = fu
            y = fu.argmax(1)
            I = y == 0
            if I.any():
                fu[I,self.y[0]] = 1
                y = fu.argmax(1)
                #assert False
        else:
            y = np.dot(S,self.y)
            y = array_functions.replace_invalid(y,self.y.min(),self.y.max())
            o.fu = y
        o.y = y
        return o

    def predict_loo(self, data):
        data = data.get_subset(data.is_labeled)
        o = Output(data)
        n = data.n
        W = self.compute_kernel(data.x, data.x)
        W[np.diag(np.ones(n)) == 1] = 0
        D = W.sum(1)
        D_inv = 1 / D
        array_functions.replace_invalid(D_inv,x_min=1,x_max=1)
        S = np.dot(np.diag(D_inv),W)

        if not data.is_regression:
            y_mat = array_functions.make_label_matrix(data.y)
            fu = np.dot(S,array_functions.try_toarray(y_mat))
            fu = array_functions.replace_invalid(fu,0,1)
            fu = array_functions.normalize_rows(fu)
            o.fu = fu
            o.y = fu.argmax(1)
        else:
            o.y = np.dot(S,data.y)
            o.fu = o.y
        return o

    def tune_loo(self, data):
        train_data = data.get_subset(data.is_train)

        param_grid = list(grid_search.ParameterGrid(self.cv_params))
        if not self.cv_params:
            return
        errors = np.empty(len(param_grid))
        for param_idx, params in enumerate(param_grid):
            self.set_params(**params)
            results = self.predict_loo(train_data)
            errors[param_idx] = results.compute_error_train(self.configs.loss_function)
        min_error = errors.min()
        best_params = param_grid[errors.argmin()]
        #print best_params
        self.set_params(**best_params)

    @property
    def prefix(self):
        return 'NW'

class ScikitLearnMethod(Method):

    _short_name_dict = {
        'Ridge': 'RidgeReg',
        'DummyClassifier': 'DumClass',
        'LogisticRegression': 'LogReg',
        'KNeighborsClassifier': 'KNN'
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

    def predict(self, data):
        assert False, 'Incorporate probabilities?'
        o = Output(data)
        o.y = self.skl_method.predict(data.x)
        return o

class SKLKNN(ScikitLearnMethod):
    def __init__(self,configs=None):
        super(SKLKNN, self).__init__(configs, neighbors.KNeighborsClassifier())
        self.cv_params['n_neighbors'] = np.asarray(list(reversed([1,3,5,15,31])))
        #self.set_params(metric=metrics.CosineDistanceMetric())
        self.set_params(algorithm='brute')

    def train(self, data):
        labeled_train = data.labeled_training_data()
        #self.skl_method.fit(array_functions.try_toarray(labeled_train.x), labeled_train.y)
        self.skl_method.fit(array_functions.try_toarray(labeled_train.x), labeled_train.y)

    def predict(self, data):
        o = Output(data)
        o.y = self.skl_method.predict(array_functions.try_toarray(data.x))
        return o

class SKLGuessClassifier(ScikitLearnMethod):
    def __init__(self,configs=None):
        assert False, 'Test this'
        super(SKLGuessClassifier, self).__init__(configs,dummy.DummyClassifier('uniform'))

class SKLMeanRegressor(ScikitLearnMethod):
    def __init__(self,configs=None):
        assert False, 'Test this'
        super(SKLMeanRegressor, self).__init__(configs,dummy.DummyRegressor('mean'))

'''
class PyQtFitMethod(Method):
    _short_name_dict = {
        'NW': 'NW'
    }

    def __init__(self,configs=MethodConfigs(),skl_method=None):
        super(PyQtFitMethod, self).__init__(configs)
        self.pyqtfit_method = nonparam_regression.NonParamRegression
        self.model = None

    def train(self, data):
        labeled_train = data.labeled_training_data()
        self.model = self.pyqtfit_method(
            labeled_train.x,
            labeled_train.y,
            method=npr_methods.SpatialAverage()
        )
        self.model.fit()
        self.model.evaluate(labeled_train.x)
        pass

    def predict(self, data):
        o = Output(data)
        assert False
        o.y = self.model.evaluate(data.x)
        return o

    def set_params(self, **kwargs):
        super(ScikitLearnMethod,self).set_params(**kwargs)

    def _pyqtfit_method_name(self):
        assert False
        return repr(self.pyqtfit_method).split('(')[0]

    @property
    def prefix(self):
        return "PyQtfit-" + PyQtFitMethod._short_name_dict[self._pyqtfit_method_name()]
'''