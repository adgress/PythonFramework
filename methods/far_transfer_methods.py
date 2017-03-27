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
import math
from timer.timer import tic, toc

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
        self.use_validation = self.configs.use_validation
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
        self.use_rbf = True
        if self.use_rbf:
            self.cv_params['C'] = self.create_cv_params(-5, 6, step, append_zero=True)
            self.cv_params['sigma_nw'] = self.create_cv_params(-5, 6, 1)
            self.cv_params['sigma_tr'] = self.create_cv_params(-5, 6, 1)
        else:
            self.cv_params['C'] = self.create_cv_params(-5, 10, step, append_zero=True)
            self.cv_params['sigma_nw'] = np.asarray([1, .5, .25, .1, .05, .025, .01])
            self.cv_params['sigma_tr'] = np.asarray([1, .5, .25, .1, .05, .025])
        self.use_prediction_graph_sparsification = False
        self.k_sparsification = 5
        self.use_oracle_graph = getattr(configs, 'use_oracle_graph', False)
        self.oracle_guidance = getattr(configs, 'oracle_guidance', None)
        self.oracle_guidance_binary = True
        self.nystrom_percentage = getattr(configs, 'nystrom_percentage', None)
        self.sigma_nw = None
        self.C = None
        self.sigma_tr = None
        self.just_nw = getattr(configs, 'just_nw', False)
        if self.just_nw:
            del self.cv_params['sigma_tr']
            del self.cv_params['C']
            self.C = 0
            self.sigma_tr = 0
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
        if not self.just_nw:
            self.source_learner.train_and_test(source_data)
            y_pred = self.source_learner.predict(target_data).y
            if self.use_oracle_graph:
                y_pred = target_data.true_y
        else:
            y_pred = np.zeros(target_data.y.shape)
        target_data.source_y_pred = y_pred
        return super(GraphTransfer, self).train_and_test(target_data)

    def train(self, data):
        I = data.is_labeled
        #y_pred_source = data.source_y_pred[I]

        y = data.y[I]
        x = data.x[I, :]
        self.transform.fit(x)
        self.x = x
        self.y = y
        #self.y_pred_source = y_pred_source

    def predict(self, data):
        # d = data_lib.Data(np.expand_dims(data.source_y_pred, 1), data.y)
        y_pred_source = data.source_y_pred
        I = np.arange(y_pred_source.size)
        if self.predict_sample is not None and self.predict_sample < y_pred_source.size:
            I = np.random.choice(y_pred_source.size, self.predict_sample, replace=False)
        if self.use_rbf:
            #L = array_functions.make_laplacian(y_pred_source[I], self.sigma_tr)
            W_source_pred = array_functions.make_rbf(y_pred_source[I], self.sigma_tr)
            if self.oracle_guidance is not None:
                y = data.true_y[I]

                n_y = y.size
                num_to_sample = math.ceil(self.oracle_guidance*n_y**2)
                rand_index1 = np.random.choice(n_y, int(num_to_sample), replace=True)
                rand_index2 = np.random.choice(n_y, int(num_to_sample), replace=True)
                if self.oracle_guidance_binary:
                    target_distances = array_functions.make_graph_distance(y)
                    distance_threshold = .2*(y.max() - y.min())
                    W_source_pred[rand_index1, rand_index2] = target_distances[rand_index1, rand_index2] <= distance_threshold
                    W_source_pred[rand_index2, rand_index1] = target_distances[rand_index2, rand_index1] <= distance_threshold
                else:
                    y_scaled = array_functions.normalize(y) * (y_pred_source.max() - y_pred_source.min())
                    W_oracle_pred = array_functions.make_rbf(y_scaled, self.sigma_tr)
                    W_source_pred[rand_index1, rand_index2] = W_oracle_pred[rand_index1, rand_index2]
                    W_source_pred[rand_index2, rand_index1] = W_oracle_pred[rand_index2, rand_index1]
            W = array_functions.make_rbf(self.transform.transform(self.x), self.sigma_nw, x2=self.transform.transform(data.x[I,:])).T

        else:
            assert self.oracle_guidance is None
            k_L = int(self.sigma_tr*I.size)
            #L = array_functions.make_laplacian_kNN(y_pred_source[I], k_L)
            W_source_pred = array_functions.make_knn(y_pred_source[I], k_L)
            k_W = int(self.sigma_nw*self.x.shape[0])
            W = array_functions.make_knn(self.transform.transform(data.x[I, :]), k_W, x2=self.transform.transform(self.x))
        if self.use_prediction_graph_sparsification:
            W_sparse = array_functions.make_knn(
                self.transform.transform(data.x[I, :]),
                self.k_sparsification,
                normalize_entries=False)
            #W_L = array_functions.make_knn(y_pred_source[I], k_L)
            W_source_pred = W_source_pred * W_sparse
        S = array_functions.make_smoothing_matrix(W)
        timing_test = False
        if self.nystrom_percentage > 0 or timing_test:
            if timing_test:
                tic()
            Sy = S.dot(self.y)
            if self.C != 0:
                lamb = 1/float(self.C)
                inv_approx = array_functions.nystrom_woodbury_laplacian(W_source_pred, lamb, self.nystrom_percentage)
                inv_approx *= lamb
                f = inv_approx.dot(Sy)
            else:
                f = Sy
            if timing_test:
                toc()
        if self.nystrom_percentage == 0 or self.nystrom_percentage is None or timing_test:
            if timing_test:
                tic()
            L = array_functions.make_laplacian_with_W(W_source_pred, normalized=False)
            A = np.eye(I.size) + self.C*L
            try:
                f = np.linalg.lstsq(A, S.dot(self.y))[0]
            except:
                print 'GraphTransferNW:predict failed, returning mean'
                f = self.y.mean() * np.ones(data.true_y.shape)
            if timing_test:
                toc()
        if timing_test:
            A_inv = np.linalg.inv(A)
            print 'approx error: ' + str(norm(inv_approx - A_inv)/norm(A_inv))
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
        if self.just_nw:
            s += '-nw'
        if getattr(self, 'predict_sample', None) is not None:
            s += '-sample=' + str(self.predict_sample)
        if self.use_rbf:
            s += '-use_rbf'
        if getattr(self, 'use_prediction_graph_sparsification', False):
            s += '-transfer_sparse=' + str(self.k_sparsification)
        if getattr(self, 'use_oracle_graph', False):
            s += '-oracle_graph'
        if getattr(self, 'oracle_guidance', None) is not None:
            if getattr(self, 'oracle_guidance_binary', False) > 0:
                s += '-guidance_binary=' + str(self.oracle_guidance)
            else:
                s += '-guidance=' + str(self.oracle_guidance)
        if getattr(self, 'nystrom_percentage', None) > 0:
            s += '-nystrom=' + str(self.nystrom_percentage)
        if getattr(self, 'use_validation', False):
            s += '-VAL'
        return s