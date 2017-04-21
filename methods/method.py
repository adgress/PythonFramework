import cvxpy as cvx
from utility.capturing import Capturing
from methods.constrained_methods import \
    PairwiseConstraint, BoundLowerConstraint, BoundUpperConstraint, \
    NeighborConstraint, BoundConstraint, HingePairwiseConstraint, EqualsConstraint, \
    SimilarConstraint, SimilarConstraintHinge, ConvexNeighborConstraint, LogisticBoundConstraint, \
    ExpNeighborConstraint
from timer import timer
from timer.timer import tic, toc

__author__ = 'Aubrey'

import abc
from copy import deepcopy

import numpy as np
from sklearn import dummy
from sklearn import grid_search
from sklearn import linear_model
from sklearn import neighbors
from sklearn.metrics import pairwise
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.gaussian_process import GaussianProcess
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import LabelEncoder


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
#from pyqt_fit import nonparam_regression
#from pyqt_fit import npr_methods
from mpipool import core as mpipool
from utility import mpi_utility
from mpi4py import MPI
import math
import random
import os
import itertools
import random
from methods import logistic_difference_optimize
from scipy import optimize
import warnings
from methods import preprocessing
try:
    from dccp.problem import is_dccp
except:
    is_dccp = lambda x: False


print_messages_cv = False


def _run_cross_validation_iteration_args(self, args):
    num_runs = 0
    #assert self.temp_dir is not None
    temp_file = None
    if self.temp_dir is not None:
        temp_file = self.temp_dir + '/' + str(args) + '.pkl'
        if os.path.isfile(temp_file):
            ret = None
            try:
                ret = helper_functions.load_object(temp_file)
                print 'Results already exist: ' + temp_file
            except:
                print 'loading file failed - deleting and rerunning...'
                os.remove(temp_file)
            if ret is not None:
                return ret
    while True:
        num_runs += 1
        if print_messages_cv:
            mpi_utility.mpi_print('CV Itr(' + str(num_runs) + '): ' + str(args), mpi_utility.get_comm())
            timer.tic()
        try:
            ret = self._run_cross_validation_iteration(args, self.curr_split, self.test_data)
            if print_messages_cv:
                timer.toc()
            if self.save_cv_temp and not helper_functions.is_laptop() and temp_file is not None:
                helper_functions.save_object(temp_file, ret)
            return ret
        except MemoryError:
            print 'Ran out of memory - restarting'
            if print_messages_cv:
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
        self.label_transform = None
        self.warm_start = False
        self.temp_dir = None
        self.save_cv_temp = True
        self.use_mpi = True
        self.use_aic = getattr(configs, 'use_aic', False)
        self.num_params = None
        self.likelihood = None
        self.include_size_in_file_name = getattr(configs, 'include_size_in_file_name', False)
        self.num_labels = getattr(configs, 'num_labels', None)
        self.preprocessor = preprocessing.IdentityPreprocessor()

    @property
    def can_use_instance_weights(self):
        return False

    def create_cv_params(self, i_low, i_high, step=1, append_zero=False, prepend_inf=False):
        a = 10**np.asarray(list(reversed(range(i_low,i_high, step))),dtype='float64')
        if append_zero:
            a = np.append(a, 0)
        if prepend_inf:
            a = np.insert(a, 0, np.inf)
        return a

    def run_pre_experiment_setup(self, data_and_splits):
        pass

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
        if hasattr(self, 'num_cv_splits'):
            num_splits = self.num_cv_splits
        perc_train = .8
        is_regression = data.is_regression
        if self.cv_use_data_type:
            splits = data_splitter.generate_splits(data.y,num_splits,perc_train,is_regression,data.is_target)
        else:
            splits = data_splitter.generate_splits(data.y,num_splits,perc_train,is_regression)
        return splits

    def _run_aic_iteration(self, params, curr_split, test_data):
        import math
        fold_results, fold_results_on_test_data = self._run_cross_validation_iteration(params, curr_split, test_data)
        fold_results.aic = 2*self.num_params - 2*math.log(self.likelihood)

    def _run_cross_validation_iteration(self, params, curr_split, test_data):   
        self.set_params(**params)
        results = self.run_method(curr_split)
        fold_results = FoldResults()
        fold_results.prediction = results
        #param_results[param_idx].set(fold_results, i)
        results_on_test_data = None
        if test_data.n > 0:
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
        #Is this necessary?
        #train_data = train_data.get_subset(train_data.is_train)
        if self.configs.use_validation:
            I = train_data.is_labeled & train_data.is_train
            #If no validation data, then unlabel random subset of training data
            allow_unlabeling = True
            if I.all() and allow_unlabeling:
                #print 'No unlabeled training for validation.  Unlabeling random subset...'
                perc_to_unlabel = .2
                to_unlabel = np.random.choice(I.size, int(np.ceil(perc_to_unlabel*I.size)), replace=False)
                train_data.y[to_unlabel] = np.nan
                I = train_data.is_labeled & train_data.is_train
            assert not I.all()
            ds = create_data_split.DataSplitter()
            splits = ds.generate_identity_split(I)
            #assert not hasattr(data, 'is_train_pairwise')
            splits[0].is_train_pairwise = getattr(data, 'is_train_pairwise', None)
        elif self.use_test_error_for_model_selection:
            I = train_data.is_train
            ds = create_data_split.DataSplitter()
            splits = ds.generate_identity_split(I)
            splits[0].is_train_pairwise = getattr(data, 'is_train_pairwise', None)
        else:
            train_data = data.get_subset(data.is_train)
            splits = self._create_cv_splits(train_data)
        assert splits[0].is_train.mean() < 1, 'No test data in CV splits!'
        data_and_splits = data_lib.SplitData(train_data,splits)
        param_grid = list(grid_search.ParameterGrid(self.cv_params))
        if not self.cv_params:
            return param_grid[0], None
        self.warm_start = False
        my_comm = mpi_utility.get_comm()
        param_results_on_test = [self.experiment_results_class(len(splits)) for i in range(len(param_grid))]
        param_results = [self.experiment_results_class(len(splits)) for i in range(len(param_grid))]
        if (my_comm is None or my_comm.Get_size() == 1 or my_comm == MPI.COMM_WORLD) \
                or not self.use_mpi:
            old_temp_dir = self.temp_dir
            for i in range(len(splits)):
                curr_split = data_and_splits.get_split(i)
                curr_split.remove_test_labels()
                self.warm_start = False
                if old_temp_dir is not None:
                    self.temp_dir = old_temp_dir + str(i)
                for param_idx, params in enumerate(param_grid):
                    #results, results_on_test = self._run_cross_validation_iteration(params, curr_split, test_data)
                    self.curr_split = curr_split
                    self.test_data = test_data
                    results, results_on_test = _run_cross_validation_iteration_args(self, params)
                    param_results[param_idx].set(results, i)
                    param_results_on_test[param_idx].set(results_on_test, i)
                    self.warm_start = True
            self.warm_start = False
            self.temp_dir = old_temp_dir
        else:
            pool = mpipool.MPIPool(comm=my_comm, debug=False, loadbalance=True, object=self)
            self.data_and_splits = pool.bcast(data_and_splits, root=0)
            self.test_data = pool.bcast(test_data, root=0)
            splits = pool.bcast(splits, root=0)
            if pool.is_master() and self.temp_dir is not None:
                helper_functions.make_dir_for_file_name(self.temp_dir)
            old_temp_dir = self.temp_dir
            for i in range(len(splits)):
                self.curr_split = data_and_splits.get_split(i)
                self.curr_split.remove_test_labels()
                if self.temp_dir is not None:
                    self.temp_dir += str(i)
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
        aggregate_test_results = data.n_test > 0
        errors_on_test_data = None
        if aggregate_test_results:
            errors_on_test_data = np.empty(len(param_grid))
        for i in range(len(param_grid)):
            agg_results = param_results[i].aggregate_error(self.configs.cv_loss_function)
            assert len(agg_results) == 1
            errors[i] = agg_results[0].mean

            if aggregate_test_results:
                agg_results_test = param_results_on_test[i].aggregate_error(self.configs.cv_loss_function)
                assert len(agg_results_test) == 1
                errors_on_test_data[i] = agg_results_test[0].mean

        min_error = errors.min()
        best_params = param_grid[errors.argmin()]
        performance_on_test_data = None
        if aggregate_test_results:
            performance_on_test_data = errors_on_test_data[errors.argmin()]
        if not self.quiet and mpi_utility.is_group_master():
            if test_data.n == 0:
                print 'No test data!'
            else:
                print best_params
                print 'CV Error: ' + str(errors.min())
                print 'Test Error: ' + str(errors_on_test_data[errors.argmin()])
        self.best_params = best_params

        return [best_params, min_error, performance_on_test_data]

    def process_data(self, data):
        data_orig = data
        data = self.preprocessor.preprocess(data, self.configs)
        labels_to_keep = np.empty(0)
        t = getattr(self.configs,'target_labels',None)
        s = getattr(self.configs,'source_labels',None)
        if t is not None and t.size > 0:
            labels_to_keep = np.concatenate((labels_to_keep,t))
        if s is not None and s.size > 0:
            s = s.ravel()
            labels_to_keep = np.concatenate((labels_to_keep,s))
            #inds = array_functions.find_set(data.y,s)
            inds = data.get_transfer_inds(s)
            data.type[inds] = data_lib.TYPE_SOURCE
            data.is_train[inds] = True
        if labels_to_keep.size > 0:
            data_old = data
            data = data.get_transfer_subset(labels_to_keep,include_unlabeled=True)
        assert data.n > 0
        return data


    def train_and_test(self, data):
        self.should_plot_g = False
        data = self.process_data(data)
        if len(self.cv_params) == 0:
            best_params = None
            min_error = None
            error_on_test_data = None
        else:
            self.running_cv = True
            best_params, min_error, error_on_test_data = self.run_cross_validation(data)
            self.set_params(**best_params)
        self.running_cv = False
        self.should_plot_g = True
        output = None
        if mpi_utility.is_group_master() or not self.use_mpi:
            output = self.run_method(data)
        comm = mpi_utility.get_comm()
        if comm != MPI.COMM_WORLD and self.use_mpi:
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

    def predict_x(self, x):
        y = np.zeros(x.shape[0])
        y[:] = np.nan
        d = data_lib.Data(x, y)
        return self.predict(d)

    def predict_loo(self, data):
        assert False, 'Not implemented!'


    def train_predict_loo(self, data):
        data = data.get_subset(data.is_labeled)
        o = Output(data)
        for i in range(data.n):
            I = array_functions.true(data.n)
            I[i] = False
            data_i = data.get_subset(I)
            self.train(data_i)
            yi = self.predict(data).y[i]
            o.y[i] = yi
            o.fu[i] = yi
        return o


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
        if configs is not None and 'metric' in configs.__dict__:
            self.metric = configs.metric
        self.instance_weights = None
        #self.metric = 'cosine'

    @property
    def can_use_instance_weights(self):
        return True

    def compute_kernel(self,x,y,bandwidth=None):
        if bandwidth is None:
            bandwidth = self.sigma
        #TODO: Optimize this for cosine similarity using cross product and matrix multiplication
        W = pairwise.pairwise_distances(x,y,self.metric)
        W = np.square(W)
        W = -bandwidth * W
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


class NadarayaWatsonKNNMethod(NadarayaWatsonMethod):
    def __init__(self,configs=MethodConfigs()):
        super(NadarayaWatsonKNNMethod, self).__init__(configs)
        self.cv_params['sigma'] = np.asarray([.5, .25, .1, .05, .025, .01])

    def compute_kernel(self,x,y):
        k = int(self.sigma*y.shape[0])
        W = array_functions.make_knn(x, k, x2=y)
        return W
        #return pairwise.rbf_kernel(x,y,self.sigma)

    @property
    def prefix(self):
        return 'NW-knn'


class ScikitLearnMethod(Method):

    _short_name_dict = {
        'Ridge': 'RidgeReg',
        'DummyClassifier': 'DumClass',
        'DummyRegressor': 'DumReg',
        'LogisticRegression': 'LogReg',
        'KNeighborsClassifier': 'KNN',
        'RidgeClassifier': 'RidgeClass'
    }

    def __init__(self,configs=MethodConfigs(),skl_method=None):
        super(ScikitLearnMethod, self).__init__(configs)
        self.skl_method = skl_method

    def can_use_instance_weights(self):
        return True

    def train(self, data):
        labeled_train = data.labeled_training_data()
        x = labeled_train.x
        if self.transform is not None:
            x = self.transform.fit_transform(x)
        self.skl_method.fit(x, labeled_train.y, labeled_train.instance_weights)

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

    @property
    def w(self):
        return self.skl_method.coef_

    @property
    def b(self):
        return self.skl_method.intercept_

    def _skl_method_name(self):
        return repr(self.skl_method).split('(')[0]

    @property
    def prefix(self):
        s = "SKL-" + ScikitLearnMethod._short_name_dict[self._skl_method_name()]
        if self.preprocessor is not None and self.preprocessor.prefix() is not None:
            s += '-' + self.preprocessor.prefix()
        return s

class SKLRidgeRegression(ScikitLearnMethod):
    def __init__(self,configs=MethodConfigs()):
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
            #d.reveal_labels(i)
            d.y[i] = d.true_y[i]
        o = Output(d)
        o.fu = y
        o.y = y
        return o

class SKLRidgeClassification(ScikitLearnMethod):
    def __init__(self, configs=MethodConfigs()):
        super(SKLRidgeClassification, self).__init__(configs, linear_model.RidgeClassifier())
        self.cv_params['alpha'] = 10 ** np.asarray(range(-8, 8), dtype='float64')
        self.set_params(alpha=0, fit_intercept=True, normalize=True, tol=1e-12)
        self.set_params(solver='auto')

        useStandardScale = True
        if useStandardScale:
            self.set_params(normalize=False)
            self.transform = StandardScaler()

class SKLLogisticRegression(ScikitLearnMethod):
    def __init__(self,configs=MethodConfigs()):
        super(SKLLogisticRegression, self).__init__(configs, linear_model.LogisticRegression())
        self.cv_params['C'] = 10**np.asarray(list(reversed(range(-5, 5))),dtype='float64')
        self.set_params(C=0,fit_intercept=True,penalty='l2')

    def predict(self, data):
        #assert False, 'Incorporate probabilities?'
        warnings.warn('Incorporate Probabilities?')
        o = Output(data)
        o.y = self.skl_method.predict(data.x)
        return o

class SKLKNN(ScikitLearnMethod):
    def __init__(self,configs=None):
        super(SKLKNN, self).__init__(configs, neighbors.KNeighborsClassifier())
        self.cv_params['n_neighbors'] = np.asarray(list(reversed([1,3,5,15,31])))
        #self.set_params(metric=metrics.CosineDistanceMetric())
        self.set_params(algorithm='brute')
        self.set_params(metric='cosine')

    def train(self, data):
        labeled_train = data.labeled_training_data()
        #self.skl_method.fit(array_functions.try_toarray(labeled_train.x), labeled_train.y)
        self.skl_method.fit(array_functions.try_toarray(labeled_train.x), labeled_train.y)

    def predict(self, data):
        o = Output(data)
        o.y = self.skl_method.predict(array_functions.try_toarray(data.x))
        return o

class SKLKNNRegression(SKLKNN):
    def __init__(self,configs=None):
        super(SKLKNN, self).__init__(configs, neighbors.KNeighborsRegressor())
        self.cv_params['n_neighbors'] = np.asarray(list(reversed([1,3,5,15,31])))
        #self.set_params(metric=metrics.CosineDistanceMetric())
        self.set_params(algorithm='brute')
        self.set_params(metric='cosine')

class SKLGuessClassifier(ScikitLearnMethod):
    def __init__(self,configs=None):
        assert False, 'Test this'
        super(SKLGuessClassifier, self).__init__(configs,dummy.DummyClassifier('uniform'))

class SKLMeanRegressor(ScikitLearnMethod):
    def __init__(self,configs=None):
        #assert False, 'Test this'
        super(SKLMeanRegressor, self).__init__(configs,dummy.DummyRegressor('mean'))

class SKLGaussianProcess(ScikitLearnMethod):
    def __init__(self,configs=None):
        #assert False, 'Test this'
        super(SKLGaussianProcess, self).__init__(configs,GaussianProcess)
        self.cv_params['nugget'] = self.create_cv_params(-5, 5)

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
    METHOD_NONPARAMETRIC = 10
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
        METHOD_CVX_NEW_CONSTRAINTS: 'cvx-constraints',
        METHOD_NONPARAMETRIC: 'NW',
    }
    def __init__(self,configs=MethodConfigs()):
        super(RelativeRegressionMethod, self).__init__(configs)
        self.can_use_test_error_for_model_selection = True
        self.cv_params['C'] = 10**np.asarray(list(reversed(range(-8,8))),dtype='float64')
        self.cv_params['C2'] = 10**np.asarray(list(reversed(range(-8,8))),dtype='float64')
        self.cv_params['C3'] = 10**np.asarray(list(reversed(range(-8,8))),dtype='float64')
        self.cv_params['C4'] = 10**np.asarray(list(reversed(range(-8,8))),dtype='float64')
        self.cv_params['s'] = 10**np.asarray(list(reversed(range(-3,3))),dtype='float64')
        self.cv_params['scale'] = 5**np.asarray(list(reversed(range(-3,3))),dtype='float64')
        self.small_param_range = configs.small_param_range
        if self.small_param_range:
            self.cv_params['C'] = 10**np.asarray(list(reversed(range(-5,5))),dtype='float64')
            self.cv_params['C2'] = 10**np.asarray(list(reversed(range(-5,5))),dtype='float64')
            self.cv_params['C3'] = 10**np.asarray(list(reversed(range(-5,5))),dtype='float64')
            self.cv_params['C4'] = 10**np.asarray(list(reversed(range(-5,5))),dtype='float64')

        self.num_features = configs.num_features
        self.use_perfect_feature_selection = configs.use_perfect_feature_selection
        self.w = None
        self.b = None
        self.transform = StandardScaler()
        '''
        if self.pca_dim > 0:
            pca = PCA(self.pca_dim,whiten=True)
            self.transform = Pipeline([('pca', pca), ('z-score', self.transform)])
        '''
        if self.num_features > 0 and not self.use_perfect_feature_selection:
            select_k_best = SelectKBest(f_regression, self.num_features)
            self.transform = Pipeline([
                ('selectK', select_k_best),
                ('z-score', self.transform),
            ])
            #self.transform = PCA(self.pca_dim,whiten=True)
        self.use_mixed_cv = configs.get('use_mixed_cv', False)
        self.use_baseline = configs.get('use_baseline', False)
        self.ridge_on_fail = configs.get('ridge_on_fail', True)
        self.tune_scale = configs.get('tune_scale', False)
        self.scipy_opt_method = configs.get('scipy_opt_method', 'L-BFGS-B')
        self.num_cv_splits = configs.get('num_cv_splits', 10)

        self.eps = configs.get('eps', 1e-6)
        self.y_transform = None
        self.y_scale_min_max = configs.get('y_scale_min_max', False)
        self.y_scale_standard = configs.get('y_scale_standard', False)
        if self.y_scale_min_max:
            self.y_transform = MinMaxScaler()
        elif self.y_scale_standard:
            self.y_transform = StandardScaler()

        self.add_random_pairwise = configs.get('use_pairwise', True)
        self.use_pairwise = self.add_random_pairwise
        self.num_pairwise = configs.get('num_pairwise', 10)
        self.pair_bound = configs.get('pair_bound', [])
        self.use_hinge = configs.get('use_hinge', False)
        self.noise_rate = configs.get('noise_rate', 0)
        self.logistic_noise = configs.get('logistic_noise', 0)
        self.use_logistic_fix = configs.get('use_logistic_fix', True)
        self.pairwise_use_scipy = configs.get('pairwise_use_scipy', True)

        self.add_random_bound = configs.get('use_bound', False)
        self.use_bound = self.add_random_bound
        self.num_bound = configs.get('num_bound', 10)
        self.use_quartiles = configs.get('use_quartiles', True)
        self.bound_logistic = configs.get('bound_logistic', True)

        self.add_random_neighbor = configs.get('use_neighbor', False)
        self.use_neighbor = self.add_random_neighbor
        self.num_neighbor = configs.get('num_neighbor', 10)
        self.use_min_pair_neighbor = configs.get('use_min_pair_neighbor', False)
        self.fast_dccp = configs.get('fast_dccp', True)
        self.init_ridge = configs.get('init_ridge', False)
        self.init_ideal = configs.get('init_ideal', False)
        self.init_ridge_train = configs.get('init_ridge_train', False)
        self.use_neighbor_logistic = configs.get('use_neighbor_logistic', False)
        self.neighbor_convex = configs.get('neighbor_convex', False)
        self.neighbor_hinge = configs.get('neighbor_hinge', False)
        self.neighbor_exp = configs.get('neighbor_exp', True)

        self.add_random_similar = configs.get('use_similar', False)
        self.use_similar = self.add_random_similar
        self.num_similar = configs.get('num_similar', False)
        self.use_similar_hinge = configs.get('use_similar_hinge', False)
        self.similar_use_scipy = configs.get('similar_use_scipy', True)

        self.keep_random_guidance = True

        self.use_test_error_for_model_selection = configs.get('use_test_error_for_model_selection', False)
        self.no_linear_term = True
        self.neg_log = False
        self.prob = None
        self.use_grad = True
        if helper_functions.is_laptop():
            self.solver = cvx.SCS
        else:
            self.solver = cvx.SCS

        self.method = RelativeRegressionMethod.METHOD_CVX_LOGISTIC_WITH_LOG
        self.method = RelativeRegressionMethod.METHOD_CVX_NEW_CONSTRAINTS

        self.w_initial = None
        self.b_initial = None
        self.optimization_failed = False
        #Why did I have this before?
        '''
        if self.use_neighbor:
            self.cv_params['C'] = 10**np.asarray(list(reversed(range(-4,4))),dtype='float64')
            self.cv_params['C4'] = 10**np.asarray(list(reversed(range(-4,4))),dtype='float64')
        '''
        if not self.use_pairwise:
            self.cv_params['C2'] = np.asarray([0])
        if not self.use_bound:
            self.cv_params['C3'] = np.asarray([0])
        if not self.use_neighbor:
            self.cv_params['C4'] = np.asarray([0])
        if not self.use_similar:
            self.cv_params['s'] = np.asarray([1])
        if not self.tune_scale:
            self.cv_params['scale'] = np.asarray([1])
        else:
            if self.use_test_error_for_model_selection:
                #self.save_cv_temp = False
                pass
            if self.use_pairwise:
                self.cv_params['C'] = 10**np.asarray(list(reversed(range(-5,5))),dtype='float64')
                self.cv_params['C2'] = 10**np.asarray(list(reversed(range(-5,5))),dtype='float64')
        if self.use_pairwise and self.use_logistic_fix:
            self.cv_params['C'] = 10**np.asarray(list(reversed(range(-5,5))),dtype='float64')
            self.cv_params['C2'] = 10**np.asarray(list(reversed(range(-5,5))),dtype='float64')
        if self.use_similar:
            self.cv_params['s'] = np.asarray([.05, .1, .2, .3],dtype='float64')
            self.cv_params['C'] = 10**np.asarray(list(reversed(range(-5,5))),dtype='float64')
            self.cv_params['C2'] = 10**np.asarray(list(reversed(range(-5,5))),dtype='float64')

        for key, values in self.cv_params.iteritems():
            if key == 's' or key == 'scale' or values.size <= 1:
                continue
            self.cv_params[key] = np.append(values, 0)


    def train_and_test(self, data):
        use_dccp = self.use_neighbor and not self.neighbor_convex and not self.neighbor_hinge
        #Solve for best w to initialize problem
        if self.y_transform is not None:
            data = deepcopy(data)
            data.true_y = self.y_transform.fit_transform(data.true_y)
            I = ~np.isnan(data.y)
            data.y[I] = self.y_transform.transform(data.y[I])

        if use_dccp and (self.init_ridge or self.init_ideal):
            new_configs = deepcopy(self.configs)
            new_configs.use_neighbor = False
            new_configs.add_random_neighbor = False
            new_configs.num_neighbor = 0
            new_instance = RelativeRegressionMethod(new_configs)
            new_instance.temp_dir = self.temp_dir
            init_data = data
            if self.init_ideal:
                init_data = deepcopy(data)
                init_data.set_train()
                init_data.set_true_y()
            r = new_instance.train_and_test(init_data)
            self.w_initial = new_instance.w
            self.b_initial = new_instance.b
        d = deepcopy(data)
        if self.num_features > 0 and self.use_perfect_feature_selection:
            assert self.num_features >= data.p, "This won't work if we already have constraints!"
            select_k_best = SelectKBest(f_regression, self.num_features)
            d.x = select_k_best.fit_transform(d.x, d.true_y)
        self.add_random_guidance(d)
        d.pairwise_ordering = None
        d.neighbor_ordering = None
        d.bound_ordering = None
        output =  super(RelativeRegressionMethod, self).train_and_test(d)
        comm = mpi_utility.get_comm()
        if comm != MPI.COMM_WORLD and self.use_mpi:
            self.optimization_failed = comm.bcast(self.optimization_failed, root=0)
        #if self.optimization_failed and (mpi_utility.is_group_master() or not self.use_mpi):
        if self.optimization_failed and self.ridge_on_fail:
            if mpi_utility.is_group_master():
                warnings.warn('Optimized failed on ' + self.split_idx_str + ' - using ridge instead...')
            self.optimization_failed = False
            c = deepcopy(self.configs)
            c.use_pairwise = False
            c.use_bound = False
            c.use_neighbor = False
            c.use_similar = False
            m = RelativeRegressionMethod(c)
            m.temp_dir = self.temp_dir  + '/ridge/'
            output = m.train_and_test(data)
        '''
        comm = mpi_utility.get_comm()
        if comm != MPI.COMM_WORLD and self.use_mpi:
            output = comm.bcast(output, root=0)
        '''
        return output


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

    def run_pre_experiment_setup(self, data_and_splits):
        super(RelativeRegressionMethod, self).run_pre_experiment_setup(data_and_splits)
        #self.add_random_guidance(data_and_splits.data)
        self.set_guidance_ordering(data_and_splits)

    def set_guidance_ordering(self, data_and_splits):
        data = data_and_splits.data
        I = data.is_train
        I = np.asarray(I.nonzero()[0])
        np.random.seed(0)
        random.seed(0)
        max_items = 2000
        if self.add_random_pairwise or self.add_random_similar:
            n = I.size
            all_pairs = [(I[i],I[j]) for i in range(n) for j in range(i+1,n)]
            all_pairs = np.asarray(all_pairs)
            rand_perm = np.random.permutation(all_pairs)
            n = min(rand_perm.shape[0], max_items)
            data.pairwise_ordering = rand_perm[0:n]
        elif self.add_random_bound:
            I = np.random.permutation(I)
            data.bound_ordering = I
        elif self.add_random_neighbor:
            triplets = set()
            values = xrange(I.shape[0])
            for i in range(max_items):
                triplet = tuple(sorted(random.sample(values,3)))
                triplets.add(triplet)
            triplets = np.asarray(list(triplets))
            data.neighbor_ordering = np.random.permutation(triplets)
            '''
            triplets = list(itertools.combinations(I,3))
            n = min(len(triplets), max_items)
            neighbors = np.asarray(random.sample(triplets, n))
            rand_perm = np.random.permutation(neighbors)
            data.neighbor_ordering = rand_perm[0:n]
            '''



    def add_random_guidance(self, data):
        num_random_types = int(self.add_random_pairwise) + int(self.add_random_bound) + \
                           int(self.add_random_neighbor) + int(self.add_random_similar)
        assert num_random_types <= 1, 'Not implemented yet'
        if getattr(data,'pairwise_relationships',None) is None:
            data.pairwise_relationships = set()
        if self.add_random_pairwise or self.add_random_similar:
            assert not self.use_baseline
            data.pairwise_relationships = set()
            #I = data.is_train & ~data.is_labeled
            I = data.is_train
            train_inds = I.nonzero()[0]
            ind_to_train_ind = {j: i for i,j in enumerate(train_inds)}
            test_func = lambda ij: True
            max_diff = data.true_y.max() - data.true_y.min()
            diff_func = lambda ij: abs(data.true_y[ij[0]] - data.true_y[ij[1]]) / max_diff
            if len(self.pair_bound) > 0:
                if len(self.pair_bound) == 1:
                    test_func = lambda ij: diff_func(ij) <= self.pair_bound[0]
                else:
                    test_func = lambda ij: self.pair_bound[0] <= diff_func(ij) <= self.pair_bound[1]
            pairwise_ordering = getattr(data, 'pairwise_ordering', None)
            num_pairs = self.num_pairwise
            if self.add_random_similar:
                num_pairs = self.num_similar
                test_func = lambda ij: diff_func(ij) <= .1
            if pairwise_ordering is None:
                sampled_pairs = array_functions.sample_pairs(I.nonzero()[0], num_pairs, test_func)
            else:
                test_func2 = lambda ij: test_func(ij) and I[ij[0]] and I[ij[1]]
                sampled_pairs = [tuple(s.tolist()) for s in pairwise_ordering if test_func2(s)]
                sampled_pairs = set(sampled_pairs[0:num_pairs])
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
                if self.add_random_pairwise:
                    if self.use_hinge:
                        constraint = HingePairwiseConstraint(x1,x2)
                    else:
                        constraint = PairwiseConstraint(x1,x2,pair[0], pair[1])
                else:
                    if self.use_similar_hinge:
                        constraint = SimilarConstraintHinge(x1,x2,max_diff)
                    else:
                        constraint = SimilarConstraint(x1,x2,max_diff)
                constraint.true_y = [data.true_y[pair[0]], data.true_y[pair[1]]]
                data.pairwise_relationships.add(constraint)
                #data.pairwise_relationships.add(pair)

        if self.add_random_bound:
            #assert False, 'Use Ordering'
            data.pairwise_relationships = set()
            bound_ordering = getattr(data, 'bound_ordering', None)
            if bound_ordering is not None:
                test_func = lambda i: data.is_train[i]
                sampled = [b for b in bound_ordering if test_func(b)]
                sampled = set(sampled[0:self.num_bound])
            else:
                I = (data.is_train).nonzero()[0]
                sampled = array_functions.sample(I, self.num_bound)
            y_median = np.percentile(data.true_y,50)
            for i in sampled:
                if self.use_quartiles:
                    if self.use_baseline:
                        data.pairwise_relationships.add(EqualsConstraint.create_quantize_constraint(data, i, 4))
                    else:
                        if self.bound_logistic:
                            cons = LogisticBoundConstraint.create_quartile_constraints(data, i)
                            data.pairwise_relationships.add(*cons)
                        else:
                            lower, upper = BoundConstraint.create_quartile_constraints(data, i)
                            lower.true_y = [data.true_y[i]]
                            upper.true_y = [data.true_y[i]]
                            data.pairwise_relationships.add(lower)
                            data.pairwise_relationships.add(upper)
                else:
                    assert not self.bound_logistic
                    if self.use_baseline:
                        data.pairwise_relationships.add(EqualsConstraint.create_quantize_constraint(data, i, 2))
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
            #assert False, 'Use Ordering'
            data.pairwise_relationships = set()
            #I = (data.is_train & ~data.is_labeled).nonzero()[0]
            I = data.is_train
            neighbor_ordering = getattr(data, 'neighbor_ordering', None)
            if neighbor_ordering is not None:
                unique_func = lambda ijk: np.unique([
                    data.true_y[ijk[0]], data.true_y[ijk[1]], data.true_y[ijk[2]]
                ]).size == 3
                test_func = lambda ijk: I[ijk[0]] and I[ijk[1]] and I[ijk[2]] and unique_func(ijk)
                '''
                for s in neighbor_ordering:
                    print str([data.true_y[s[0]], data.true_y[s[1]], data.true_y[s[2]]])
                '''
                sampled = [tuple(s.tolist()) for s in neighbor_ordering if test_func(s)]
                sampled = set(sampled[0:self.num_neighbor])
            else:
                assert False, 'Make sure all Y are unique'
                I = (data.is_train).nonzero()[0]
                sampled = array_functions.sample_n_tuples(I, self.num_neighbor, 3, True)
            for i in sampled:
                i1,i2,i3 = i
                y1,y2,y3 = data.true_y[[i1,i2,i3]]
                if self.use_min_pair_neighbor:
                    ordering = helper_functions.compute_min_pair(y1,y2,y3)
                    triplet = tuple(i[j] for j in ordering)
                elif self.neighbor_convex or self.neighbor_exp:
                    #ordering = helper_functions.compute_min_pair(y1,y2,y3)
                    #triplet = tuple(i[j] for j in ordering)
                    values = data.true_y[list(i)]
                    sorted_idx = np.argsort(values)
                    y1,y2,y3 = data.true_y[[i[k] for k in sorted_idx]]
                    if np.abs(y2-y1) < np.abs(y2-y3):
                        idx = (sorted_idx[1], sorted_idx[0], sorted_idx[2])
                        triplet = tuple(i[j] for j in idx)
                    else:
                        triplet = tuple(i[j] for j in sorted_idx)
                else:
                    if np.abs(y1-y2) < np.abs(y1-y3):
                        triplet = i
                    else:
                        triplet = (i1,i3,i2)

                x1,x2,x3 = data.x[triplet,:]
                y1,y2,y3 = data.true_y[list(triplet)]
                if self.neighbor_convex:
                    constraint = ConvexNeighborConstraint(x1,x2,x3)
                elif self.neighbor_exp:
                    constraint = ExpNeighborConstraint(x1,x2,x3)
                else:
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
        if self.num_features > 0:
            dim_to_use = min(self.num_features, x.shape[0] - 1)

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
            method = self.scipy_opt_method
            options = {
                'disp': False
            }
            w0 = np.zeros(x.shape[1]+1)
            constraints = []
            if self.use_bound and self.bound_logistic and not self.use_baseline:
                x_bound, bounds = LogisticBoundConstraint.generate_bounds_for_scipy_optimize(
                    data.pairwise_relationships,
                    self.transform
                )
                opt_data = logistic_difference_optimize.optimize_data(
                    x, y, self.C, self.C3
                )
                opt_data.eps = self.eps
                opt_data.x_bound = x_bound
                opt_data.bounds = bounds
                eval = logistic_difference_optimize.logistic_bound.create_eval(opt_data)
                grad = logistic_difference_optimize.logistic_bound.create_grad(opt_data)
            elif self.use_neighbor and self.neighbor_convex and not self.neighbor_hinge and not self.neighbor_exp:
                method = 'SLSQP'
                x_neighbor,x_low,x_high = ConvexNeighborConstraint.generate_neighbors_for_scipy_optimize(
                    data.pairwise_relationships,
                    self.transform
                )
                f1 = logistic_difference_optimize.logistic_neighbor.create_constraint_neighbor(x_low, x_high)
                f2 = logistic_difference_optimize.logistic_neighbor.create_constraint_neighbor2(x_neighbor, x_low, x_high)
                constraints = [{
                    'type': 'ineq',
                    'fun': f1
                }]
                constraints += [{
                    'type': 'ineq',
                    'fun': f2
                }]

                C = self.C
                C4 = self.C4
                opt_data = logistic_difference_optimize.optimize_data(
                    x, y, C, C4
                )
                opt_data_feasible = logistic_difference_optimize.optimize_data(
                    x, y, C, 0
                )
                opt_data.x_neighbor = x_neighbor
                opt_data.x_low = x_low
                opt_data.x_high = x_high
                opt_data.eps = self.eps
                opt_data_feasible.x_neighbor = x_neighbor
                opt_data_feasible.x_low = x_low
                opt_data_feasible.x_high = x_high
                eval = logistic_difference_optimize.logistic_neighbor.create_eval(opt_data)
                grad = logistic_difference_optimize.logistic_neighbor.create_grad(opt_data)
                '''
                eval_feasible = logistic_difference_optimize.logistic_neighbor.create_eval(opt_data_feasible)
                grad_feasible = logistic_difference_optimize.logistic_neighbor.create_grad(opt_data_feasible)
                if not self.use_grad:
                    grad_feasible = None
                with Capturing() as output:
                    feasible_results = optimize.minimize(eval_feasible,w0,method=method,jac=grad_feasible,options=options,constraints=constraints)
                w0_feasible = feasible_results.x
                w0 = w0_feasible
                '''
            elif self.use_pairwise and self.pairwise_use_scipy and not self.use_hinge:
                x_low,x_high,_,_ = PairwiseConstraint.generate_pairs_for_scipy_optimize(
                    data.pairwise_relationships,
                    self.transform
                )
                #self.C = 10
                #self.C2 = 10
                C = self.C
                C2 = self.C2
                #C2 = 0
                #C = 0

                opt_data = logistic_difference_optimize.optimize_data(
                    x, y, C, C2
                )
                opt_data.x_low = x_low
                opt_data.x_high = x_high
                eval = logistic_difference_optimize.logistic_pairwise.create_eval(opt_data)
                grad = logistic_difference_optimize.logistic_pairwise.create_grad(opt_data)
            elif self.use_similar and self.similar_use_scipy and not self.use_similar_hinge:
                x1,x2 = PairwiseConstraint.generate_pairs_for_scipy_optimize(
                    data.pairwise_relationships,
                    self.transform
                )
                C = self.C
                C2 = self.C2
                s = self.s
                opt_data = logistic_difference_optimize.optimize_data(
                    x, y, C, C2
                )
                opt_data.s = s
                opt_data.x1 = x1
                opt_data.x2 = x2
                opt_data.eps = self.eps
                eval = logistic_difference_optimize.logistic_similar.create_eval(opt_data)
                grad = logistic_difference_optimize.logistic_similar.create_grad(opt_data)
            else:
                self.solve_cvx(x, y, data)
                return
            if not self.use_grad:
                grad = None
            options['maxiter'] = 1000
            opt_data.s = self.s
            opt_data.scale = self.scale
            #tic()
            with Capturing() as output:
                results = optimize.minimize(eval,w0,method=method,jac=grad,options=options,constraints=constraints)
            w1 = results.x
            #toc()
            #options['gtol'] = 1e-3
            compare_results = False
            if compare_results or not self.running_cv:
                results2 = optimize.minimize(eval,w0,method=method,jac=None,options=options,constraints=constraints)
                w2 = results2.x

                from numpy.linalg import norm
                if norm(results2.x) == 0:
                    print 'Error: Norm is 0'
                else:
                    err = norm(results.x-results2.x)/norm(results.x)
                    print 'eval vs. jac Error: ' + str(err)
                '''
                #tic()
                self.solve_cvx(x, y, data)
                #toc()
                w_cvx = self.w
                b_cvx = self.b
                print 'Error cvx: ' + str(norm(results2.x[0:-1] - w_cvx.T)/norm(w_cvx))
                '''
                if results2.success:
                    print 'Results2 Success'
                    pass
                print results.message
            if not np.isfinite(w1).all():
                w1[:] = 0
            if not results.success:
                if self.ridge_on_fail:
                    w1[:] = 0
                    if not self.running_cv:
                        self.optimization_failed = True
                if not self.running_cv:
                    import warnings
                    warnings.warn('Optimization failed!')

            self.w, self.b = logistic_difference_optimize.unpack_linear(w1)
            y_train_pred = x.dot(self.w) + self.b
            pass

    def solve_cvx(self, x, y, data):
        p = x.shape[1]
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
        constraints = []
        if self.use_pairwise and not self.use_hinge:
            pairwise_reg2 = PairwiseConstraint.generate_cvx(train_pairwise, func, transform=self.transform, scale=self.s)
        elif self.use_bound and not self.bound_logistic and False:
            bound_reg3 = BoundConstraint.generate_cvx(train_pairwise, func, transform=self.transform, scale=self.s)
        elif self.use_neighbor and self.neighbor_hinge and self.neighbor_convex and False:
            neighbor_reg4 = ConvexNeighborConstraint.generate_cvx(train_pairwise, func, transform=self.transform, scale=self.s)
        elif self.use_neighbor and self.neighbor_exp:
            neighbor_reg4, constraints = ExpNeighborConstraint.generate_cvx(train_pairwise, func, transform=self.transform, scale=self.s)
        else:
            for c in train_pairwise:
                cons = []
                c2 = deepcopy(c)
                c2.transform(self.transform)
                if c2.is_pairwise():
                    val, cons = c2.to_cvx(func, scale=self.s)
                    pairwise_reg2 += val
                    #assert False
                elif c2.is_tertiary():
                    if not c2.use_dccp():
                        val, cons = c2.to_cvx(func, scale=self.s)
                        neighbor_reg4 += val
                else:
                    val, cons = c2.to_cvx(func)
                    bound_reg3 += val
                constraints += cons
        if self.add_random_neighbor and not self.neighbor_convex and not self.neighbor_exp:
            neighbor_reg4, t, t_constraints = NeighborConstraint.to_cvx_dccp(
                train_pairwise,
                func,
                self.use_neighbor_logistic
            )
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
        elif self.init_ridge_train:
            new_configs = deepcopy(self.configs)
            new_configs.use_neighbor = False
            new_configs.add_random_neighbor = False
            new_configs.num_neighbor = 0
            new_configs.init_ridge_train = False
            new_instance = RelativeRegressionMethod(new_configs)
            new_instance.temp_dir = self.temp_dir
            new_instance.save_cv_temp = False
            new_instance.use_mpi = False
            r = new_instance.train_and_test(data)
            w.value = new_instance.w
            b.value = new_instance.b
        else:
            w.value = None
            b.value = None
        #assert prob.is_dcp()
        if not prob.is_dcp():
            assert is_dccp(prob)
        print_messages = False
        if print_messages:
            timer.tic()
        params = [self.C_param.value, self.C2_param.value, self.C3_param.value, self.C4_param.value]
        #ret = prob.solve(method = 'dccp',solver = 'MOSEK')
        #ret = prob.solve(method = 'dccp')
        try:
            with Capturing() as output:
                #ret = prob.solve(cvx.ECOS, False, {'warm_start': warm_start})
                if prob.is_dcp():
                    ret = prob.solve(self.solver, True, {'warm_start': warm_start})
                    #ret = prob.solve(cvx.CVXOPT, False, {'warm_start': warm_start})
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

                    w.value = self.w_initial
                    b.value = self.b_initial
                    ret = prob.solve(solver=self.solver, **options)
                    saved_w = w.value
                    saved_b = b.value

                    '''
                    w.value = self.w_initial
                    b.value = self.b_initial
                    max_iter = options['max_iter']
                    options['max_iter'] = 1
                    for i in range(max_iter):
                        options['tau'] *= options['mu']
                        ret2 = prob.solve(solver=self.solver, **options)
                    '''
            w_value = w.value
            b_value = b.value
            #print prob.status
            assert w_value is not None and b_value is not None
            #print a.value
            #print b.value
        except Exception as e:
            print str(e) + ': ' + str(params)
            #print 'cvx status: ' + str(prob.status)
            k = 0
            w_value = k*np.zeros((p,1))
            b_value = 0
        if print_messages:
            print 'params: ' + str(params)
            timer.toc()
        if warm_start:
            self.prob = prob
        '''
        for c in train_pairwise:
            c.transform(self.transform)
            c_cvx,_ = c.to_cvx(func)
            print c_cvx.value
            if np.isnan(c_cvx.value):
                c.to_cvx(func)
        '''
        self.w = w_value
        self.b = b_value

    def predict(self, data):
        o = RelativeRegressionOutput(data)
        if data.n == 0:
            return o
        if self.method == RelativeRegressionMethod.METHOD_RIDGE_SURROGATE:
            assert False, 'y_transform?'
            o = self.ridge_reg.predict(data)
        else:
            x = self.transform.transform(data.x)
            y = x.dot(self.w) + self.b
            if self.y_transform is not None:
                y = self.y_transform.inverse_transform(y)
                o.true_y = self.y_transform.inverse_transform(o.true_y)
            o.fu = y
            o.y = y
            #f = lambda x: x.dot(self.w)[0,0] + self.b
            f = lambda x: x.dot(self.w)+ self.b
            n = len(data.pairwise_relationships)
            is_pairwise_correct = array_functions.false(n)
            for i, c in enumerate(data.pairwise_relationships):
                c2 = deepcopy(c)
                c2.transform(self.transform)
                is_pairwise_correct[i] = c2.is_correct(f)
            o.is_pairwise_correct = is_pairwise_correct
        if not np.isfinite(o.fu).all():
            print 'weird prediction'
        #p = self.transform.named_steps['pca']
        #z = self.transform.named_steps['z-score']
        return o

    @property
    def prefix(self):
        s = 'RelReg'
        if self.method != RelativeRegressionMethod.METHOD_CVX:
            s += '-' + RelativeRegressionMethod.METHOD_NAMES[self.method]
        use_pairwise = self.use_pairwise
        use_bound = getattr(self, 'use_bound', False)
        use_neighbor = getattr(self, 'use_neighbor', False)
        use_baseline = getattr(self, 'use_baseline', False)
        use_similar = getattr(self, 'use_similar', False)
        using_cvx = False
        if getattr(self, 'use_nw', False):
            s += '-NW'
        if not use_pairwise and not use_bound and not use_neighbor and not use_similar:
            using_cvx = True
            s += '-noPairwiseReg'
        else:
            if use_pairwise:
                #if self.num_pairwise > 0 and self.use_pairwise:
                if self.use_pairwise:
                    if getattr(self, 'use_hinge', False):
                        s += '-numRandPairsHinge=' + str(int(self.num_pairwise))
                        using_cvx = True
                    else:
                        s += '-numRandPairs=' + str(int(self.num_pairwise))
                        if getattr(self, 'pairwise_use_scipy', False):
                            s += '-scipy'
                            if not getattr(self, 'use_grad', True):
                                s += '-noGrad'
                    pair_bound = getattr(self, 'pair_bound', ())
                    try:
                        pair_bound = tuple(pair_bound)
                    except:
                        pair_bound = (pair_bound,)
                    if len(pair_bound) > 0 and \
                        not (len(pair_bound) == 1 and pair_bound[0] == 1):
                        s += '-pairBound=' + str(pair_bound)
                if use_baseline:
                    s += '-baseline'
                #if self.no_linear_term:
                #    s += '-noLinear'
                if self.neg_log:
                    s += '-negLog'
                if not use_baseline:
                    if getattr(self, 'noise_rate', 0) > 0:
                        s += '-noise=' + str(self.noise_rate)
                    if getattr(self, 'logistic_noise', 0) > 0:
                        s += '-logNoise=' + str(self.logistic_noise)
                if getattr(self, 'use_logistic_fix', False):
                    s += '-logFix'
            if use_bound:
                hasRandBounds = self.num_bound > 0 and self.add_random_bound
                using_cvx = True
                if getattr(self, 'bound_logistic', False):
                    s += '-numRandLogBounds=' + str(int(self.num_bound))
                    using_cvx = False
                elif getattr(self, 'use_quartiles', False):
                    s += '-numRandQuartiles=' + str(int(self.num_bound))
                else:
                    s += '-numRandBound=' + str(int(self.num_bound))
                if use_baseline:
                    s += '-baseline'
            if use_neighbor and self.num_neighbor > 0 and self.add_random_neighbor:
                use_convex = getattr(self, 'neighbor_convex', False)
                use_exp = getattr(self, 'neighbor_exp', False)
                using_cvx = False
                if getattr(self, 'use_min_pair_neighbor', False):
                    s += '-numMinNeighbor=' + str(int(self.num_neighbor))
                elif use_convex or use_exp:
                    if use_exp:
                        s += '-numRandNeighborExp=' + str(int(self.num_neighbor))
                        using_cvx = True
                    elif getattr(self, 'neighbor_hinge', False):
                        s += '-numRandNeighborConvexHinge=' + str(int(self.num_neighbor))
                        using_cvx = True
                    else:
                        s += '-numRandNeighborConvex=' + str(int(self.num_neighbor))
                else:
                    s += '-numRandNeighbor=' + str(int(self.num_neighbor))
                if not use_convex and not use_exp:
                    if getattr(self, 'fast_dccp', False):
                        s += '-fastDCCP'
                    if getattr(self, 'init_ideal', False):
                        s += '-init_ideal'
                    if getattr(self, 'init_ridge', False):
                        s += '-initRidge'
                    if getattr(self, 'init_ridge_train', False):
                        s += '-initRidgeTrain'
                    if getattr(self, 'use_neighbor_logistic', False):
                        s += '-logistic'
            if use_similar and self.num_similar > 0 and self.add_random_similar:
                if self.use_similar_hinge:
                    using_cvx = True
                    s += '-numSimilarHinge=' + str(int(self.num_similar))
                else:
                    s += '-numSimilar=' + str(int(self.num_similar))
                    if getattr(self, 'similar_use_scipy', False):
                        s += '-scipy'
            if getattr(self, 'use_mixed_cv', False):
                s += '-mixedCV'
            if not getattr(self, 'ridge_on_fail', True) and not using_cvx:
                s += '-noRidgeOnFail'
            if getattr(self, 'tune_scale', False) and not using_cvx:
                s += '-tuneScale'
            if getattr(self, 'small_param_range', False):
                s += '-smallScale'
            if (use_similar or use_bound or use_neighbor) and not using_cvx:
                eps = getattr(self,'eps',logistic_difference_optimize.eps)
                if eps is not None and eps != logistic_difference_optimize.eps:
                    s += '-eps=' + str(eps)
            if hasattr(self, 'solver'):
                s += '-solver=' + str(self.solver)

        if getattr(self, 'y_scale_min_max', False):
            s += '-minMax'
        elif getattr(self, 'y_scale_standard', False):
            s += '-zScore'
        num_features = getattr(self,'num_features', -1)
        if num_features  > 0:
            if getattr(self, 'use_perfect_feature_selection', False):
                s += '-numFeatsPerfect=' + str(num_features)
            else:
                s += '-numFeats=' + str(num_features)
        if not using_cvx and getattr(self, 'scipy_opt_method', 'BFGS') != 'BFGS':
            s += '-' + self.scipy_opt_method
        if self.use_test_error_for_model_selection:
            s += '-TEST'
        elif getattr(self, 'num_cv_splits', 5) != 5:
            s += '-nCV=' + str(self.num_cv_splits)
        if getattr(self, 'include_size_in_file_name', False):
            assert len(self.num_labels) == 1
            s += '-num_labels=' + str(self.num_labels)
        return s


class NonparametricRelativeRegressionMethod(RelativeRegressionMethod):
    def __init__(self, configs=MethodConfigs()):
        super(NonparametricRelativeRegressionMethod, self).__init__(configs)
        self.cv_params = dict()
        self.cv_params['C2'] =  10 ** np.asarray(list(reversed(range(-8, 10))), dtype='float64')
        self.cv_params['C2'][-1] = 0
        self.cv_params['sigma'] = 10 ** np.asarray(list(reversed(range(-5, 5))), dtype='float64')
        self.method = RelativeRegressionMethod.METHOD_NONPARAMETRIC
        self.ridge_on_fail = False
        self.f = None
        self.nw_learner = NadarayaWatsonMethod(configs)
        assert self.use_pairwise

    def train_and_test(self, data):
        return super(NonparametricRelativeRegressionMethod, self).train_and_test(data)


    def train(self, data):
        labeled_train = data.labeled_training_data()
        x = labeled_train.x
        y = labeled_train.y
        x = self.transform.fit_transform(x, y)
        if self.num_features > 0:
            dim_to_use = min(self.num_features, x.shape[0] - 1)

        x_low, x_high, inds_low, inds_high = PairwiseConstraint.generate_pairs_for_scipy_optimize(
            data.pairwise_relationships,
            self.transform
        )

        x_all = self.transform.transform(data.x)
        for i, j in enumerate(inds_low):
            assert (x_low[i] == x_all[j]).all()
        for i, j in enumerate(inds_high):
            assert (x_high[i] == x_all[j]).all()

        C2 = self.C2
        sigma = self.sigma

        S = array_functions.make_rbf(x_all, sigma, x2=x)

        opt_data = logistic_difference_optimize.optimize_data(
            None, y, 0, C2
        )
        opt_data.S = S
        opt_data.inds_low = inds_low
        opt_data.inds_high = inds_high

        eval = logistic_difference_optimize.logistic_pairwise_nonparametric.create_eval(opt_data)
        #grad = logistic_difference_optimize.logistic_pairwise_nonparametric.create_grad(opt_data)
        grad = None

        options = {
            'disp': False,
            'maxfun': np.inf
        }
        if not self.use_grad:
            grad = None
        options['maxiter'] = 1000
        constraints = []
        f0 = np.zeros((x_all.shape[0], 1))
        #with Capturing() as output:
        if not self.running_cv:
            pass
        results = optimize.minimize(eval, f0, method=self.scipy_opt_method, jac=grad, options=options, constraints=constraints)
        data_copy = deepcopy(data)
        data_copy.pairwise_relationships = None
        if results.success:
            self.f = results.x
            data_copy.y = self.f
            data_copy.true_y = self.f
            self.nw_learner.train_and_test(data_copy)
        else:
            print 'Error'
        self.nw_learner.train_and_test(data_copy)

    def predict(self, data):
        o = RelativeRegressionOutput(data)
        o.y = o.fu = self.nw_learner.predict(data).y
        return o


