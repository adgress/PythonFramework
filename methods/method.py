import cvxpy as cvx
from utility.capturing import Capturing
from methods.constrained_methods import \
    PairwiseConstraint, BoundLowerConstraint, BoundUpperConstraint, \
    NeighborConstraint, BoundConstraint, HingePairwiseConstraint
from timer import timer

__author__ = 'Aubrey'

import abc
from copy import deepcopy

import numpy as np
from sklearn import dummy
from sklearn import grid_search
from sklearn import linear_model
from sklearn import neighbors
from sklearn.metrics import pairwise
from sklearn.preprocessing import StandardScaler

from configs.base_configs import MethodConfigs
from data import data as data_lib
from data_sets import create_data_split
from results_class import results as results_lib
from results_class.results import FoldResults, Output
from results_class.results import Output, RelativeRegressionOutput
from saveable.saveable import Saveable
from utility import array_functions
from utility import helper_functions
#from dccp.dccp_problem import is_dccp
from dccp.problem import is_dccp
#from pyqt_fit import nonparam_regression
#from pyqt_fit import npr_methods
from mpipool import core as mpipool
from utility import mpi_utility
from mpi4py import MPI
import math
import random
import os




def _run_cross_validation_iteration_args(self, args):
    num_runs = 0
    assert self.temp_dir is not None
    temp_file = self.temp_dir + '/' + str(args) + '.pkl'
    if os.path.isfile(temp_file):
        ret = helper_functions.load_object(temp_file)
        print 'Results already exist: ' + temp_file
        return ret
    while True:
        num_runs += 1
        mpi_utility.mpi_print('CV Itr(' + str(num_runs) + '): ' + str(args), mpi_utility.get_comm())
        timer.tic()
        try:
            ret = self._run_cross_validation_iteration(args, self.curr_split, self.test_data)
            timer.toc()
            helper_functions.save_object(temp_file, ret)
            return ret
        except MemoryError:
            print 'Ran out of memory - restarting'
            timer.toc()
        else:
            assert False, 'Some other error occured'



class Method(Saveable):

    def __init__(self,configs=MethodConfigs()):
        super(Method, self).__init__(configs)
        self._params = []
        self.cv_params = {}
        self.is_classifier = True
        self.experiment_results_class = results_lib.ExperimentResults
        self.cv_use_data_type = True
        self.use_test_error_for_model_selection = False
        self.can_use_test_error_for_model_selection = False
        self._estimated_error = None
        self.quiet = True
        self.best_params = None
        self.transform = None
        self.warm_start = False
        self.temp_dir = None

    def create_cv_params(self, i_low, i_high):
        return 10**np.asarray(list(reversed(range(i_low,i_high))),dtype='float64')

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
        num_splits = 5
        if hasattr(self, 'num_splits'):
            num_splits = self.num_splits
        perc_train = .8
        is_regression = data.is_regression
        if self.cv_use_data_type:
            splits = data_splitter.generate_splits(data.y,num_splits,perc_train,is_regression,data.is_target)
        else:
            splits = data_splitter.generate_splits(data.y,num_splits,perc_train,is_regression)
        return splits

    def _run_cross_validation_iteration(self, params, curr_split, test_data):   
        self.set_params(**params)
        results = self.run_method(curr_split)
        fold_results = FoldResults()
        fold_results.prediction = results
        #param_results[param_idx].set(fold_results, i)
        results_on_test_data = self.predict(test_data)
        fold_results_on_test_data = FoldResults()
        fold_results_on_test_data.prediction = results_on_test_data
        #param_results_on_test[param_idx].set(fold_results_on_test_data, i)
        return fold_results, fold_results_on_test_data

    def run_cross_validation(self,data):
        assert data.n_train_labeled > 0
        train_data = deepcopy(data)
        #test_data = data.get_subset(data.is_test)
        test_data = data.get_test_data()
        if self.configs.use_validation:
            I = train_data.is_labeled
            train_data.reveal_labels()
            ds = create_data_split.DataSplitter()
            splits = ds.generate_identity_split(I)
        elif self.use_test_error_for_model_selection:
            I = train_data.is_train
            ds = create_data_split.DataSplitter()
            splits = ds.generate_identity_split(I)
            splits[0].is_train_pairwise = getattr(data, 'is_train_pairwise', None)
        else:
            train_data = data.get_subset(data.is_train)
            splits = self._create_cv_splits(train_data)
        data_and_splits = data_lib.SplitData(train_data,splits)
        param_grid = list(grid_search.ParameterGrid(self.cv_params))
        if not self.cv_params:
            return param_grid[0], None
        self.warm_start = False
        my_comm = mpi_utility.get_comm()
        param_results_on_test = [self.experiment_results_class(len(splits)) for i in range(len(param_grid))]
        param_results = [self.experiment_results_class(len(splits)) for i in range(len(param_grid))]
        if my_comm is None or my_comm.Get_size() == 1 or my_comm == MPI.COMM_WORLD:
            #Results when using test data to do model selection

            #Results when using cross validation
            for i in range(len(splits)):
                curr_split = data_and_splits.get_split(i)
                curr_split.remove_test_labels()
                self.warm_start = False
                for param_idx, params in enumerate(param_grid):
                    results, results_on_test = self._run_cross_validation_iteration(params, curr_split, test_data)
                    param_results[param_idx].set(results, i)
                    param_results_on_test[param_idx].set(results_on_test, i)
                    self.warm_start = True
            self.warm_start = False
        else:
            pool = mpipool.MPIPool(comm=my_comm, debug=False, loadbalance=True, object=self)
            self.data_and_splits = pool.bcast(data_and_splits, root=0)
            self.test_data = pool.bcast(test_data, root=0)
            splits = pool.bcast(splits, root=0)
            if pool.is_master():
                helper_functions.make_dir_for_file_name(self.temp_dir)
            old_temp_dir = self.temp_dir
            for i in range(len(splits)):
                self.curr_split = data_and_splits.get_split(i)
                self.curr_split.remove_test_labels()
                self.temp_dir += str(i) + '-'
                if pool.is_master():
                    helper_functions.make_dir_for_file_name(self.temp_dir)
                all_split_results = pool.map(_run_cross_validation_iteration_args, param_grid)
                self.temp_dir = old_temp_dir
                pool.close()
                if pool.is_master():
                    param_idx = 0
                    for split_results, split_results_on_test in all_split_results:                        
                        param_results[param_idx].set(split_results, i)
                        param_results_on_test[param_idx].set(split_results, i)
                        param_idx = param_idx + 1                    
                del self.curr_split
            if pool.is_master():
                helper_functions.delete_dir_if_exists(self.temp_dir)
            del self.data_and_splits
            del self.test_data                        
            param_results = pool.bcast(param_results, root=0)
            param_results_on_test = pool.bcast(param_results_on_test, root=0)            

        errors = np.empty(len(param_grid))
        errors_on_test_data = np.empty(len(param_grid))
        for i in range(len(param_grid)):
            agg_results = param_results[i].aggregate_error(self.configs.cv_loss_function)
            assert len(agg_results) == 1
            errors[i] = agg_results[0].mean

            agg_results_test = param_results_on_test[i].aggregate_error(self.configs.cv_loss_function)
            assert len(agg_results_test) == 1
            errors_on_test_data[i] = agg_results_test[0].mean

        min_error = errors.min()
        best_params = param_grid[errors.argmin()]
        if not self.quiet and mpi_utility.is_group_master():
            print best_params
        self.best_params = best_params
        return [best_params, min_error, errors_on_test_data[errors.argmin()]]

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
        if len(self.cv_params) == 0:
            best_params = None
            min_error = None
            error_on_test_data = None
        else:
            best_params, min_error, error_on_test_data = self.run_cross_validation(data)
            self.set_params(**best_params)
        self.should_plot_g = True
        output = None
        if mpi_utility.is_group_master():
            output = self.run_method(data)
        comm = mpi_utility.get_comm()
        if comm != MPI.COMM_WORLD:
            output = comm.bcast(output, root=0)
        f = FoldResults()
        f.prediction = output
        f.estimated_error = min_error
        f.error_on_test_data = error_on_test_data
        self.estimated_error = min_error
        self.error_on_test_data = error_on_test_data
        if best_params is not None:
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
        'DummyRegressor': 'DumReg',
        'LogisticRegression': 'LogReg',
        'KNeighborsClassifier': 'KNN',
    }

    def __init__(self,configs=MethodConfigs(),skl_method=None):
        super(ScikitLearnMethod, self).__init__(configs)
        self.skl_method = skl_method

    def train(self, data):
        labeled_train = data.labeled_training_data()
        x = labeled_train.x
        if self.transform is not None:
            x = self.transform.fit_transform(x)
        self.skl_method.fit(x, labeled_train.y)

    def predict(self, data):
        o = Output(data)
        x = data.x
        if self.transform is not None:
            x = self.transform.transform(x)
        o.y = self.skl_method.predict(x)
        o.y = array_functions.vec_to_2d(o.y)
        o.fu = o.y
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
        self.set_params(alpha=0,fit_intercept=True,normalize=True,tol=1e-12)
        self.set_params(solver='auto')

        useStandardScale = True
        if useStandardScale:
            self.set_params(normalize=False)
            self.transform = StandardScaler()

    def predict_loo(self, data):
        d = data.get_subset(data.is_train & data.is_labeled)
        y = np.zeros(d.n)
        for i in range(d.n):
            xi = d.x[i,:]
            d.y[i] = np.nan
            self.train(d)
            o_i = self.predict(d)
            y[i] = o_i.y[i]
            d.reveal_labels(i)
        o = Output(d)
        o.fu = y
        o.y = y
        return o

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
        #assert False, 'Test this'
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


class RelativeRegressionMethod(Method):
    METHOD_ANALYTIC = 1
    METHOD_CVX = 2
    METHOD_RIDGE = 3
    METHOD_RIDGE_SURROGATE = 4
    METHOD_CVX_LOGISTIC = 5
    METHOD_CVX_LOGISTIC_WITH_LOG = 6
    METHOD_CVX_LOGISTIC_WITH_LOG_NEG = 7
    METHOD_CVX_LOGISTIC_WITH_LOG_SCALE = 8
    METHOD_CVX_NEW_CONSTRAINTS = 9
    CVX_METHODS = {
        METHOD_CVX,
        METHOD_CVX_LOGISTIC,
        METHOD_CVX_LOGISTIC_WITH_LOG,
        METHOD_CVX_LOGISTIC_WITH_LOG_NEG,
        METHOD_CVX_LOGISTIC_WITH_LOG_SCALE,
        METHOD_CVX_NEW_CONSTRAINTS
    }
    CVX_METHODS_LOGISTIC = {
        METHOD_CVX_LOGISTIC,
        METHOD_CVX_LOGISTIC_WITH_LOG,
        METHOD_CVX_LOGISTIC_WITH_LOG_NEG,
        METHOD_CVX_LOGISTIC_WITH_LOG_SCALE
    }
    CVX_METHODS_LOGISTIC_WITH_LOG = {
        METHOD_CVX_LOGISTIC_WITH_LOG,
        METHOD_CVX_LOGISTIC_WITH_LOG_NEG,
        METHOD_CVX_LOGISTIC_WITH_LOG_SCALE
    }
    METHOD_NAMES = {
        METHOD_ANALYTIC: 'analytic',
        METHOD_CVX: 'cvx',
        METHOD_RIDGE: 'ridge',
        METHOD_RIDGE_SURROGATE: 'ridge-surr',
        METHOD_CVX_LOGISTIC: 'cvx-log',
        METHOD_CVX_LOGISTIC_WITH_LOG: 'cvx-log-with-log',
        METHOD_CVX_LOGISTIC_WITH_LOG_NEG: 'cvx-log-with-log-neg',
        METHOD_CVX_LOGISTIC_WITH_LOG_SCALE: 'cvx-log-with-log-scale',
        METHOD_CVX_NEW_CONSTRAINTS: 'cvx-constraints'
    }
    def __init__(self,configs=MethodConfigs()):
        super(RelativeRegressionMethod, self).__init__(configs)
        self.can_use_test_error_for_model_selection = True
        self.cv_params['C'] = 10**np.asarray(list(reversed(range(-8,8))),dtype='float64')
        self.cv_params['C2'] = 10**np.asarray(list(reversed(range(-8,8))),dtype='float64')
        self.cv_params['C3'] = 10**np.asarray(list(reversed(range(-8,8))),dtype='float64')
        self.cv_params['C4'] = 10**np.asarray(list(reversed(range(-8,8))),dtype='float64')

        self.w = None
        self.b = None
        self.transform = StandardScaler()

        self.use_mixed_cv = configs.use_mixed_cv

        self.add_random_pairwise = configs.use_pairwise
        self.use_pairwise = configs.use_pairwise
        self.num_pairwise = configs.num_pairwise
        self.pair_bound = configs.pair_bound
        self.use_hinge = configs.use_hinge
        self.noise_rate = configs.noise_rate
        self.logistic_noise = configs.logistic_noise

        self.add_random_bound = configs.use_bound
        self.use_bound = configs.use_bound
        self.num_bound = configs.num_bound
        self.use_quartiles = configs.use_quartiles

        self.add_random_neighbor = configs.use_neighbor
        self.use_neighbor = configs.use_neighbor
        self.num_neighbor = configs.num_neighbor
        self.use_min_pair_neighbor = configs.use_min_pair_neighbor
        self.fast_dccp = configs.fast_dccp
        self.init_ridge = configs.init_ridge

        self.use_test_error_for_model_selection = configs.use_test_error_for_model_selection
        self.no_linear_term = True
        self.neg_log = False
        self.prob = None

        if helper_functions.is_laptop():
            self.solver = cvx.SCS
        else:
            self.solver = cvx.SCS

        self.method = RelativeRegressionMethod.METHOD_CVX_LOGISTIC_WITH_LOG
        self.method = RelativeRegressionMethod.METHOD_CVX_NEW_CONSTRAINTS

        self.w_initial = None
        self.b_initial = None
        if self.use_neighbor:
            self.cv_params['C'] = 10**np.asarray(list(reversed(range(-4,4))),dtype='float64')
            self.cv_params['C4'] = 10**np.asarray(list(reversed(range(-4,4))),dtype='float64')
        if not self.use_pairwise:
            self.cv_params['C2'] = np.asarray([0])
        if not self.use_bound:
            self.cv_params['C3'] = np.asarray([0])
        if not self.use_neighbor:
            self.cv_params['C4'] = np.asarray([0])

    def train_and_test(self, data):
        use_dccp = self.use_neighbor

        #Solve for best w to initialize problem
        if use_dccp and self.init_ridge:
            new_configs = deepcopy(self.configs)
            new_configs.use_neighbor = False
            new_configs.add_random_neighbor = False
            new_configs.num_neighbor = 0
            new_instance = RelativeRegressionMethod(new_configs)
            r = new_instance.train_and_test(data)
            self.w_initial = new_instance.w
            self.b_initial = new_instance.b
        self.add_random_guidance(data)
        return super(RelativeRegressionMethod, self).train_and_test(data)

    def _create_cv_splits(self,data):
        splits = super(RelativeRegressionMethod, self)._create_cv_splits(data)
        perc_test = .2
        if not self.use_mixed_cv:
            perc_test = 0
        n = len(data.pairwise_relationships)
        num_test = math.floor(perc_test*n)
        for i in range(len(splits)):
            is_train_pairwise = array_functions.true(n)
            if n > 0 and num_test > 0:
                is_train_pairwise[array_functions.sample(n, num_test)] = False
            splits[i].is_train_pairwise = is_train_pairwise
        return splits

    def add_random_guidance(self, data):
        num_random_types = int(self.add_random_pairwise) + int(self.add_random_bound) + int(self.add_random_neighbor)
        assert num_random_types <= 1, 'Not implemented yet'
        if self.add_random_pairwise:
            data.pairwise_relationships = set()
            I = data.is_train & ~data.is_labeled
            test_func = None
            if len(self.pair_bound) > 0:
                diff = data.true_y.max() - data.true_y.min()
                diff_func = lambda ij: abs(data.true_y[ij[0]] - data.true_y[ij[1]]) / diff
                if len(self.pair_bound) == 1:
                    test_func = lambda ij: diff_func(ij) <= self.pair_bound[0]
                else:
                    test_func = lambda ij: self.pair_bound[0] <= diff_func(ij) <= self.pair_bound[1]
            sampled_pairs = array_functions.sample_pairs(I.nonzero()[0], self.num_pairwise, test_func)
            for i,j in sampled_pairs:
                pair = (i,j)
                diff = data.true_y[j] - data.true_y[i]
                if self.logistic_noise > 0:
                    diff += np.random.logistic(scale=self.logistic_noise)
                if diff <= 0:
                    pair = (j,i)
                #data.pairwise_relationships.add(pair)
                x1 = data.x[pair[0],:]
                x2 = data.x[pair[1],:]
                if self.use_hinge:
                    constraint = HingePairwiseConstraint(x1,x2)
                else:
                    constraint = PairwiseConstraint(x1,x2)
                constraint.true_y = [data.true_y[pair[0]], data.true_y[pair[1]]]
                data.pairwise_relationships.add(constraint)
                #data.pairwise_relationships.add(pair)
        if self.add_random_bound:
            data.pairwise_relationships = set()
            I = (data.is_train & ~data.is_labeled).nonzero()[0]
            sampled = array_functions.sample(I, self.num_bound)
            y_median = np.percentile(data.true_y,50)
            for i in sampled:
                if self.use_quartiles:
                    lower, upper = BoundConstraint.create_quartile_constraints(data, i)
                    lower.true_y = [data.true_y[i]]
                    upper.true_y = [data.true_y[i]]
                    data.pairwise_relationships.add(lower)
                    data.pairwise_relationships.add(upper)
                else:
                    xi = data.x[i,:]
                    yi_true = data.true_y[i]
                    if yi_true > y_median:
                        constraint = BoundLowerConstraint(xi, y_median)
                    elif yi_true < y_median:
                        constraint = BoundUpperConstraint(xi, y_median)
                    else:
                        continue
                    constraint.true_y = [yi_true]
                    data.pairwise_relationships.add(constraint)
        if self.add_random_neighbor:
            data.pairwise_relationships = set()
            I = (data.is_train & ~data.is_labeled).nonzero()[0]
            sampled = array_functions.sample_n_tuples(I, self.num_neighbor, 3, True)
            for i in sampled:
                i1,i2,i3 = i
                y1,y2,y3 = data.true_y[[i1,i2,i3]]
                if self.use_min_pair_neighbor:
                    ordering = helper_functions.compute_min_pair(y1,y2,y3)
                    triplet = tuple(i[j] for j in ordering)
                else:
                    if np.abs(y1-y2) < np.abs(y1-y3):
                        triplet = i
                    else:
                        triplet = (i1,i3,i2)

                x1,x2,x3 = data.x[triplet,:]
                constraint = NeighborConstraint(x1,x2,x3)
                constraint.true_y = [data.true_y[a] for a in triplet]
                data.pairwise_relationships.add(constraint)
        for s in data.pairwise_relationships:
            if random.random() <= self.noise_rate:
                s.flip()
        data.pairwise_relationships = np.asarray(list(data.pairwise_relationships))
        data.is_train_pairwise = array_functions.true(data.pairwise_relationships.size)

    def train(self, data):
        is_labeled_train = data.is_train & data.is_labeled
        labeled_train = data.labeled_training_data()
        x = labeled_train.x
        y = labeled_train.y
        x_orig = x
        x = self.transform.fit_transform(x, y)

        use_ridge = self.method in {
            RelativeRegressionMethod.METHOD_RIDGE,
            RelativeRegressionMethod.METHOD_RIDGE_SURROGATE
        }
        n, p = x.shape
        if use_ridge:
            ridge_reg = SKLRidgeRegression(self.configs)
            ridge_reg.set_params(alpha=self.C)
            ridge_reg.set_params(normalize=False)
            '''
            d = deepcopy(data)
            d.x[is_labeled_train,:] = x
            ridge_reg.train(d)
            '''
            ridge_reg.train(data)
            w_ridge = array_functions.vec_to_2d(ridge_reg.skl_method.coef_)
            b_ridge = ridge_reg.skl_method.intercept_
            self.w = w_ridge
            self.b = b_ridge
            self.ridge_reg = ridge_reg
        elif self.method == RelativeRegressionMethod.METHOD_ANALYTIC:
            x_bias = np.hstack((x,np.ones((n,1))))
            A = np.eye(p+1)
            A[p,p] = 0
            XX = x_bias.T.dot(x_bias)
            v = np.linalg.lstsq(XX + self.C*A,x_bias.T.dot(y))
            w_anal = array_functions.vec_to_2d(v[0][0:p])
            b_anal = v[0][p]
            self.w = w_anal
            self.b = b_anal
        elif self.method in RelativeRegressionMethod.CVX_METHODS:
            w = cvx.Variable(p)
            b = cvx.Variable(1)
            loss = cvx.sum_entries(
                cvx.power(
                    x*w + b - y,
                    2
                )
            )
            reg = cvx.norm(w)**2
            pairwise_reg2 = 0
            bound_reg3 = 0
            neighbor_reg4 = 0
            assert self.no_linear_term

            assert self.method == RelativeRegressionMethod.METHOD_CVX_NEW_CONSTRAINTS
            func = lambda x:x*w + b
            t_constraints = []
            train_pairwise = data.pairwise_relationships[data.is_train_pairwise]
            for c in train_pairwise:
                c2 = deepcopy(c)
                c2.transform(self.transform)
                if c2.is_pairwise():
                    pairwise_reg2 += c2.to_cvx(func)
                elif c2.is_tertiary():
                    pass
                    #neighbor_reg4 += c.to_cvx(func)
                else:
                    bound_reg3 += c2.to_cvx(func)
            if self.add_random_neighbor:
                neighbor_reg4, t, t_constraints = NeighborConstraint.to_cvx_dccp(train_pairwise, func)
            warm_start = self.prob is not None and self.warm_start
            warm_start = False
            if warm_start:
                prob = self.prob
                self.C_param.value = self.C
                self.C2_param.value = self.C2
                self.C3_param.value = self.C3
                self.C4_param.value = self.C4
                w = self.w_var
                b = self.b_var
            else:
                constraints = []
                self.C_param = cvx.Parameter(sign='positive', value=self.C)
                self.C2_param = cvx.Parameter(sign='positive', value=self.C2)
                self.C3_param = cvx.Parameter(sign='positive', value=self.C3)
                self.C4_param = cvx.Parameter(sign='positive', value=self.C4)
                obj = cvx.Minimize(loss + self.C_param*reg +
                                   self.C2_param*pairwise_reg2 +
                                   self.C3_param*bound_reg3 +
                                   self.C4_param*neighbor_reg4)
                prob = cvx.Problem(obj,constraints + t_constraints)
                #prob = cvx.Problem(obj,constraints)
                self.w_var = w
                self.b_var = b
            if self.init_ridge:
                w.value = self.w_initial
                b.value = self.b_initial
            else:
                w.value = None
                b.value = None
            #assert prob.is_dcp()
            if not prob.is_dcp():
                assert is_dccp(prob)
            print_messages = True
            if print_messages:
                timer.tic()
            params = [self.C_param.value, self.C2_param.value, self.C3_param.value, self.C4_param.value]
            #ret = prob.solve(method = 'dccp',solver = 'MOSEK')
            #ret = prob.solve(method = 'dccp')
            try:
                with Capturing() as output:
                    #ret = prob.solve(cvx.ECOS, False, {'warm_start': warm_start})
                    if prob.is_dcp():
                        ret = prob.solve(self.solver, False, {'warm_start': warm_start})
                    else:
                        print str(params)
                        options = {
                            'method': 'dccp'
                        }
                        if self.fast_dccp:
                            options = {
                                'method': 'dccp',
                                'max_iter': 20,
                                'tau': .25,
                                'mu': 2,
                                'tau_max': 1e6
                            }
                        '''
                        ret = prob.solve(solver=self.solver, **options)
                        saved_w = w.value
                        saved_b = b.value
                        '''

                        max_iter = options['max_iter']
                        options['max_iter'] = 1
                        w.value = self.w_initial
                        b.value = self.b_initial
                        for i in range(max_iter):
                            options['tau'] *= options['mu']
                            ret2 = prob.solve(solver=self.solver, **options)

                w_value = w.value
                b_value = b.value
                #print prob.status
                assert w_value is not None and b_value is not None
                #print a.value
                #print b.value
            except Exception as e:
                print str(e) + ':' + str(params)
                #print 'cvx status: ' + str(prob.status)
                k = 0
                w_value = k*np.zeros((p,1))
                b_value = 0
            if print_messages:
                print 'params: ' + str(params)
                timer.toc()
            if warm_start:
                self.prob = prob
            self.w = w_value
            self.b = b_value
            '''
            obj2 = cvx.Minimize(loss + self.C*reg)
            try:
                prob2 = cvx.Problem(obj2, constraints)
                prob2.solve()
                w2 = w.value
                b2 = b.value
                print 'b error: ' + str(array_functions.relative_error(b_value,b2))
                print 'w error: ' + str(array_functions.relative_error(w_value,w2))
                print 'pairwise_reg value: ' + str(pairwise_reg.value)
            except:
                pass
            '''
        '''
        print 'w rel error: ' + str(array_functions.relative_error(w_value,w_ridge))
        #print 'b rel error: ' + str(array_functions.relative_error(b_value,b_ridge))

        print 'w analytic rel error: ' + str(array_functions.relative_error(w_value,w_anal))
        #print 'b analytic rel error: ' + str(array_functions.relative_error(b_value,b_anal))
        print 'w norm: ' + str(norm(w_value))
        print 'w analytic norm: ' + str(norm(w_anal))
        print 'w ridge norm: ' + str(norm(w_ridge))
        assert self.b is not None
        '''

    def predict(self, data):
        o = RelativeRegressionOutput(data)

        if self.method == RelativeRegressionMethod.METHOD_RIDGE_SURROGATE:
            o = self.ridge_reg.predict(data)
        else:
            x = self.transform.transform(data.x)
            y = x.dot(self.w) + self.b
            o.fu = y
            o.y = y
            f = lambda x: x.dot(self.w)[0,0] + self.b
            n = data.pairwise_relationships.size
            is_pairwise_correct = array_functions.false(n)
            for i, c in enumerate(data.pairwise_relationships):
                c2 = deepcopy(c)
                c2.transform(self.transform)
                is_pairwise_correct[i] = c2.is_correct(f)
            o.is_pairwise_correct = is_pairwise_correct
        return o

    @property
    def prefix(self):
        s = 'RelReg'
        if self.method != RelativeRegressionMethod.METHOD_CVX:
            s += '-' + RelativeRegressionMethod.METHOD_NAMES[self.method]
        use_pairwise = self.use_pairwise
        use_bound = getattr(self, 'use_bound', False)
        use_neighbor = getattr(self, 'use_neighbor', False)
        if not use_pairwise and not use_bound and not use_neighbor:
            s += '-noPairwiseReg'
        else:
            if use_pairwise:
                if self.num_pairwise > 0 and self.add_random_pairwise:
                    if getattr(self, 'use_hinge', False):
                        s += '-numRandPairsHinge=' + str(int(self.num_pairwise))
                    else:
                        s += '-numRandPairs=' + str(int(self.num_pairwise))
                    pair_bound = getattr(self, 'pair_bound', ())
                    try:
                        pair_bound = tuple(pair_bound)
                    except:
                        pair_bound = (pair_bound,)
                    if len(pair_bound) > 0 and \
                        not (len(pair_bound) == 1 and pair_bound[0] == 1):
                        s += '-pairBound=' + str(pair_bound)
                #if self.no_linear_term:
                #    s += '-noLinear'
                if self.neg_log:
                    s += '-negLog'
            if use_bound:
                hasRandBounds = self.num_bound > 0 and self.add_random_bound
                if getattr(self, 'use_quartiles', False):
                    s += '-numRandQuartiles=' + str(int(self.num_bound))
                else:
                    s += '-numRandBound=' + str(int(self.num_bound))
            if use_neighbor and self.num_neighbor > 0 and self.add_random_neighbor:
                if getattr(self, 'use_min_pair_neighbor', False):
                    s += '-numMinNeighbor=' + str(int(self.num_neighbor))
                else:
                    s += '-numRandNeighbor=' + str(int(self.num_neighbor))
                if getattr(self, 'fast_dccp', False):
                    s += '-fastDCCP'
                if getattr(self, 'init_ridge', False):
                    s += '-initRidge'
            if getattr(self, 'use_mixed_cv', False):
                s += '-mixedCV'
            if getattr(self, 'noise_rate', 0) > 0:
                s += '-noise=' + str(self.noise_rate)
            if getattr(self, 'logistic_noise', 0) > 0:
                s += '-logNoise=' + str(self.logistic_noise)
            if hasattr(self, 'solver'):
                s += '-solver=' + str(self.solver)
        if self.use_test_error_for_model_selection:
            s += '-TEST'
        return s
