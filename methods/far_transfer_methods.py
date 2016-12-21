import copy
from copy import deepcopy
import numpy as np
from numpy.linalg import norm
import method
from preprocessing import NanLabelEncoding, NanLabelBinarizer
from data import data as data_lib
from sklearn.neighbors import KernelDensity
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from results_class.results import Output
import cvxpy as cvx
import scipy
import transfer_methods
from utility import array_functions
from results_class import results

class GraphTransfer(method.Method):
    def __init__(self, configs=None):
        super(GraphTransfer, self).__init__(configs)
        self.cv_params = dict()
        self.alpha = 0
        self.just_transfer = getattr(configs, 'just_transfer', False)
        self.just_target = getattr(configs, 'just_target', False)
        self.cv_params['alpha'] = [0, .2, .4, .6, .8, 1]
        configs = deepcopy(configs)
        self.source_learner = method.NadarayaWatsonMethod(deepcopy(configs))

        self.nw_transfer = method.NadarayaWatsonMethod(deepcopy(configs))
        self.nw_target = method.NadarayaWatsonMethod(deepcopy(configs))
        self.nw_target.quiet = True
        assert not (self.just_transfer and self.just_target)
        if self.just_transfer:
            del self.cv_params['alpha']
            self.alpha = 0
        if self.just_target:
            del self.cv_params['alpha']
            self.alpha = 1


    def train_and_test(self, data):
        self.source_learner.use_validation = self.use_validation
        self.nw_transfer.use_validation = self.use_validation
        self.nw_target.use_validation = self.use_validation
        if self.just_transfer:
            del self.cv_params['alpha']
            self.alpha = 0
        if self.just_target:
            del self.cv_params['alpha']
            self.alpha = 1
        is_source = data.data_set_ids == self.configs.source_labels[0]
        is_target = data.data_set_ids == self.configs.target_labels[0]
        source_data = data.get_subset(is_source)
        target_data = data.get_subset(is_target)
        source_data.y = source_data.true_y
        source_data.set_target()
        source_data.data_set_ids[:] = self.configs.target_labels[0]
        self.source_learner.train_and_test(source_data)
        y_pred = self.source_learner.predict(target_data).y
        target_data.source_y_pred = y_pred
        return super(GraphTransfer, self).train_and_test(target_data)

    def train(self, data):
        assert data.is_target.all()
        I = data.is_labeled
        #Need unlabeled data if using validation for hyperparameter tuning
        I[:] = True
        y_pred_source = data.source_y_pred[I]

        transfer_data = data.get_subset(I)
        transfer_data.x = np.expand_dims(y_pred_source, 1)
        self.nw_transfer.train_and_test(transfer_data)
        target_data = data.get_subset(I)
        self.nw_target.train_and_test(target_data)
        x = 0

    def predict(self, data):
        #d = data_lib.Data(np.expand_dims(data.source_y_pred, 1), data.y)
        if not self.running_cv:
            #array_functions.plot_2d_sub(data.x, data.true_y, data_set_ids=data.data_set_ids, title=None, sizes=10)
            x = 0
        transfer_data = deepcopy(data)
        transfer_data.x = np.expand_dims(data.source_y_pred, 1)
        transfer_pred =  self.nw_transfer.predict(transfer_data)
        target_pred = self.nw_target.predict(data)
        alpha = self.alpha
        if self.just_target:
            alpha = 1
        elif self.just_transfer:
            alpha = 0
        transfer_pred.y = (1-alpha)*transfer_pred.y + alpha*target_pred.y
        transfer_pred.fu = (1-alpha)*transfer_pred.fu + alpha*target_pred.fu
        return transfer_pred

    @property
    def prefix(self):
        s = 'GraphTransfer'
        if self.just_transfer:
            s += '_tr'
        if getattr(self, 'just_target', False):
            s += '_ta'
        if getattr(self, 'use_validation', False):
            s += '-VAL'
        return s

class GraphTransferNW(GraphTransfer):
    def __init__(self, configs=None):
        super(GraphTransferNW, self).__init__(configs)
        self.cv_params = dict()
        step = 2
        self.cv_params['C'] = self.create_cv_params(-5, 6, step, append_zero=True)
        '''
        self.cv_params['sigma_nw'] = self.create_cv_params(-5, 6, 1)
        self.cv_params['sigma_tr'] = self.create_cv_params(-9, 6, 1)
        '''
        self.cv_params['sigma_nw'] = np.asarray([20, 10, 5, 3])
        self.cv_params['sigma_tr'] = np.asarray([20, 10, 5, 3])
        self.sigma_nw = None
        self.C = None
        self.sigma_tr = None
        configs = deepcopy(configs)
        self.source_learner = method.NadarayaWatsonMethod(deepcopy(configs))
        self.source_learner.configs.source_labels = None
        self.source_learner.configs.target_labels = None
        self.transform = StandardScaler()
        self.predict_sample = None
        self.use_validation = getattr(configs, 'use_validation', False)
        self.nw_learner = method.NadarayaWatsonMethod(deepcopy(configs))
        self.nw_learner.configs.source_labels = None
        self.nw_learner.configs.target_labels = None
        self.source_learner.configs.use_validation = False
        self.nw_learner.configs.use_validation = False
        self.sampled_inds = None

    def train_and_test(self, data):
        is_source = data.data_set_ids == self.configs.source_labels[0]
        is_target = data.data_set_ids == self.configs.target_labels[0]
        source_data = data.get_subset(is_source)
        target_data = data.get_subset(is_target)
        source_data.y = source_data.true_y
        source_data.set_target()
        source_data.data_set_ids[:] = self.configs.target_labels[0]
        self.source_learner.train_and_test(source_data)
        y_pred = self.source_learner.predict(target_data).y
        target_data.source_y_pred = y_pred
        return super(GraphTransfer, self).train_and_test(target_data)

    def train(self, data):
        I = data.is_labeled
        y_pred_source = data.source_y_pred[I]
        y = data.y[I]
        x = data.x[I, :]
        self.transform.fit(x)
        self.x = x
        self.y = y
        self.y_pred_source = y_pred_source

    def predict(self, data):
        # d = data_lib.Data(np.expand_dims(data.source_y_pred, 1), data.y)
        y_pred_source = data.source_y_pred
        I = np.arange(y_pred_source.size)
        if self.predict_sample is not None and self.predict_sample < y_pred_source.size:
            I = np.random.choice(y_pred_source.size, self.predict_sample, replace=False)
        #L = array_functions.make_laplacian(y_pred_source[I], self.sigma_tr)
        #W = array_functions.make_rbf(self.transform.transform(self.x), self.sigma_nw, x2=self.transform.transform(data.x[I,:])).T
        L = array_functions.make_laplacian_kNN(y_pred_source[I], self.sigma_tr)
        W = array_functions.make_knn(self.transform.transform(self.x), self.sigma_nw, x2=self.transform.transform(data.x[I,:])).T
        S = array_functions.make_smoothing_matrix(W)

        A = np.eye(I.size) + self.C*L
        try:
            f = np.linalg.lstsq(A, S.dot(self.y))[0]
        except:
            print 'GraphTransferNW:predict failed, returning mean'
            f = self.y.mean() * np.ones(data.true_y.shape)

        o = results.Output(data)
        if self.predict_sample is not None:
            nw_data = data_lib.Data(data.x[I,:], f)
            self.nw_learner.train_and_test(nw_data)
            nw_output = self.nw_learner.predict(data)
            o.y = nw_output.y
            o.fu = nw_output.y
        else:
            o.y = f
            o.fu = f

        return o

    @property
    def prefix(self):
        s = 'GraphTransferNW'
        if getattr(self, 'predict_sample', None) is not None:
            s += '-sample=' + str(self.predict_sample)
        if getattr(self, 'use_validation', False):
            s += '-VAL'
        return s