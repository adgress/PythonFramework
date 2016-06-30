__author__ = 'Aubrey'
import copy
from copy import deepcopy
import numpy as np

import method
from preprocessing import NanLabelEncoding
from data import data as data_lib
from sklearn.neighbors import KernelDensity
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from results_class.results import Output
import cvxpy as cvx
import scipy

class TargetTranfer(method.Method):
    def __init__(self, configs=None):
        super(TargetTranfer, self).__init__(configs)
        self.base_learner = method.SKLLogisticRegression(configs)
        self.cv_params = {}
        self.base_learner.experiment_results_class = self.experiment_results_class

    def train(self, data):
        self.base_learner.train_and_test(data)

    def train_and_test(self, data):
        #data_copy2 = self._prepare_data(data,include_unlabeled=True)
        #results2 = super(TargetTranfer, self).train_and_test(data_copy2)
        data_copy = self._prepare_data(data,include_unlabeled=True)
        #data_copy = data_copy.get_with_labels(self.configs.target_labels)
        #data_copy = data_copy.get_transfer_subset(self.configs.target_labels, include_unlabeled=True)
        results = super(TargetTranfer, self).train_and_test(data_copy)
        #a = results.prediction.fu - results2.prediction.fu[data_copy2.is_labeled,:]
        #print str(a.any())
        return results

    def _prepare_data(self, data, include_unlabeled=True):
        target_labels = self.configs.target_labels
        data_copy = data.get_transfer_subset(target_labels,include_unlabeled=include_unlabeled)

        data_copy = data_copy.get_subset(data_copy.is_target)
        is_source = ~data_copy.has_true_label(target_labels)
        data_copy.type[is_source] = data_lib.TYPE_SOURCE
        data_copy.is_train[is_source] = True
        #data_copy = data.get_with_labels(target_labels)
        return data_copy

    def predict(self, data):
        o = self.base_learner.predict(data)
        if self.label_transform is not None:
            o.true_y = self.label_transform.transform(o.true_y)
        return o

    @method.Method.estimated_error.getter
    def estimated_error(self):
        return self.base_learner.estimated_error

    @property
    def prefix(self):
        return 'TargetTransfer+' + self.base_learner.prefix

class FuseTransfer(TargetTranfer):
    def __init__(self, configs=None):
        super(FuseTransfer, self).__init__(configs)
        self.use_oracle = False
        #self.target_weight_scale = None
        self.target_weight_scale = .75
        self.label_transform = NanLabelEncoding()

    def train(self, data):
        is_labeled_train = data.is_labeled & data.is_train
        n_labeled_target = (data.is_train & is_labeled_train).sum()
        n_labeled_source = (data.is_train & is_labeled_train).sum()
        data.instance_weights = np.ones(data.n)
        if self.target_weight_scale is not None:
            assert 0 <= self.target_weight_scale <= 1
            data.instance_weights[data.is_source] /= n_labeled_source
            data.instance_weights[data.is_target] /= n_labeled_target
            data.instance_weights[data.is_target] *= self.target_weight_scale
            data.instance_weights[data.is_source] *= (1-self.target_weight_scale)
        y_old = data.y
        if self.label_transform is not None:
            data.y = self.label_transform.fit_transform(data.y)
        super(FuseTransfer, self).train(data)
        data.y = y_old

    def _prepare_data(self, data,include_unlabeled=True):
        source_labels = self.configs.source_labels
        target_labels = self.configs.target_labels
        data_copy = copy.deepcopy(data)
        #source_inds = array_functions.find_set(data_copy.true_y,source_labels)
        if self.use_oracle:
            oracle_labels = self.configs.oracle_labels
            data_copy = data_copy.get_transfer_subset(
                np.concatenate((oracle_labels.ravel(),target_labels.ravel())),
                include_unlabeled=True
            )
        data_copy.data_set_ids = np.zeros(data_copy.n)
        for i, s in enumerate(source_labels):
            source_inds = data_copy.get_transfer_inds(s)
            if not data_copy.is_regression:
                data_copy.change_labels(s, target_labels)
            data_copy.type[source_inds] = data_lib.TYPE_SOURCE
            data_copy.is_train[source_inds] = True
            data_copy.data_set_ids[source_inds] = i+1
        data_copy.reveal_labels(data_copy.is_source)
        return data_copy

    @property
    def prefix(self):
        s = 'FuseTransfer+' + self.base_learner.prefix
        if 'target_weight_scale' in self.__dict__ and self.target_weight_scale is not None:
            s += '-tws=' + str(self.target_weight_scale)
        if 'use_oracle' in self.__dict__ and self.use_oracle:
            s += '-Oracle'
        return s


class HypothesisTransfer(FuseTransfer):
    def __init__(self, configs=None):
        super(HypothesisTransfer, self).__init__(configs)
        self.cv_params = {
            'C': self.create_cv_params(-5,5),
            'C2': self.create_cv_params(-5, 5),
            'C3': self.create_cv_params(-5, 5),
        }
        self.w = None
        self.b = None
        self.base_source_learner = method.SKLRidgeClassification(deepcopy(configs))
        self.base_source_learner.cv_use_data_type = False
        self.source_w = []
        self.transform = StandardScaler()
        self.use_oracle = False
        self.just_target = True


    def train_and_test(self, data):
        source_labels = self.configs.source_labels
        data = self._prepare_data(data)
        for i, s in enumerate(source_labels):
            #source_inds = data.get_transfer_inds(s)
            source_data = data.get_subset(data.data_set_ids == i+1)
            self.base_source_learner.train_and_test(source_data)
            w = np.squeeze(self.base_source_learner.w)
            w /= np.linalg.norm(w)
            b = self.base_source_learner.b
            self.source_w.append(w)
            pass
        target_labels = self.configs.target_labels
        target_data = data.get_subset(data.data_set_ids == 0)
        return super(HypothesisTransfer, self).train_and_test(target_data)

    def train(self, data):
        x = self.transform.fit_transform(data.x[data.is_labeled & data.is_train])
        y = data.y[data.is_labeled]
        y = self.label_transform.fit_transform(y)
        n = y.size
        p = data.p
        self.b = y.mean()
        c = cvx.Variable(len(self.source_w))
        ws1 = self.source_w[0]
        ws2 = self.source_w[1]

        constraints = [c >= 0]
        if self.just_target:
            constraints.append(c[1] == 0)
        loss = 0
        for i in range(y.size):
            xi = x[i,:]
            yi = y[i]
            x_mi = np.delete(x, i, axis=0)
            y_mi = np.delete(y, i, axis=0)
            b_mi = y_mi.mean() / (n - 1) ** 2
            A = x_mi.T.dot(x_mi) - (self.C + self.C2)*np.eye(p)
            k = x_mi.T.dot(y_mi) + x_mi.T.sum(1)*b_mi - self.C2*(ws1*c[0] + ws2*c[1])
            #w_mi = np.linalg.solve(A, k)
            w_mi = scipy.linalg.inv(A)*k
            loss += cvx.power(w_mi.T*xi + b_mi - yi,2)
        reg = cvx.power(cvx.norm2(c),2)
        obj = cvx.Minimize(loss + self.C3*reg)
        prob = cvx.Problem(obj, constraints)
        assert prob.is_dcp()
        try:
            prob.solve(cvx.SCS)
            c_value = np.asarray(c.value)
        except Exception as e:
            print str(e)
            c_value = np.zeros(p)

        A = x.T.dot(x) + (self.C + self.C2)*np.eye(p)
        k = x.T.dot(y) + x.T.sum(1)*self.b - self.C2*(ws1*c_value[0] + ws2*c_value[1])
        self.w = np.linalg.solve(A, k)
        pass

    def predict(self, data):
        o = Output(data)
        x = self.transform.transform(data.x)
        y = x.dot(self.w) + self.b
        #y = np.round(y)
        y[y >= .5] = 1
        y[y < .5] = 0
        o.y = y
        o.fu = y
        if self.label_transform is not None:
            o.true_y = self.label_transform.transform(o.true_y)
        return o

    @property
    def prefix(self):
        s = 'HypTransfer'
        if getattr(self, 'just_target', False):
            s += '-target'
        return s


class ModelSelectionTransfer(method.ModelSelectionMethod):
    def __init__(self, configs=None):
        super(ModelSelectionTransfer, self).__init__(configs)
        self.methods.append(TargetTranfer(configs))
        self.methods.append(FuseTransfer(configs))
        for m in self.methods:
            m.base_learner = method.NadarayaWatsonMethod(configs)

    @property
    def prefix(self):
        return 'ModelSelTransfer'

class ReweightedTransfer(method.Method):
    def __init__(self, configs=None):
        super(ReweightedTransfer, self).__init__(configs)
        self.target_kde = None
        self.source_kde = None
        self.kde_bandwidths = 10**np.asarray(range(-6,6),dtype='float64')
        c = deepcopy(configs)
        c.temp_dir = None
        self.base_learner = method.NadarayaWatsonMethod(configs)
        self.cv_params = {
            'B': np.asarray([2, 4, 8, 16, 32])
        }
        self.base_learner_cv_keys = []

    def train_and_test(self, data):
        assert self.base_learner.can_use_instance_weights
        target_data = data.get_transfer_subset(self.configs.target_labels.ravel(),include_unlabeled=False)
        source_data = data.get_transfer_subset(self.configs.source_labels.ravel(), include_unlabeled=False)
        is_source = data.get_transfer_inds(self.configs.source_labels.ravel())
        data.type[is_source] = data_lib.TYPE_SOURCE

        x_T = target_data.x
        x_S = source_data.x

        params = {'bandwidth': self.kde_bandwidths}
        grid = GridSearchCV(KernelDensity(), params)
        grid.fit(x_T)
        self.target_kde = deepcopy(grid.best_estimator_)
        grid.fit(x_S)
        self.source_kde = deepcopy(grid.best_estimator_)

        old_cv = self.cv_params.copy()
        old_base_cv = self.base_learner.cv_params.copy()

        assert set(old_cv.keys()) & set(old_base_cv.keys()) == set()
        self.cv_params.update(self.base_learner.cv_params)
        self.base_learner_cv_keys = old_base_cv.keys()

        o = super(ReweightedTransfer, self).train_and_test(data)
        self.cv_params = old_cv
        self.base_learner.cv_params = old_base_cv
        return o

    def train(self, data):
        I = data.is_labeled
        weights = self.get_weights(data.x)
        assert np.all(weights >=0 )
        weights[weights > self.B] = self.B
        data.instance_weights = weights
        for key in self.base_learner_cv_keys:
            setattr(self.base_learner, key, getattr(self, key))
        self.base_learner.train(data)

    def get_weights(self, x):
        target_scores = np.exp(self.target_kde.score_samples(x))
        source_scores = np.exp(self.source_kde.score_samples(x))
        return target_scores / source_scores

    def predict(self, data):
        data.instance_weights = self.get_weights(data.x)
        return self.base_learner.predict(data)

    @property
    def prefix(self):
        return 'CovShift'

