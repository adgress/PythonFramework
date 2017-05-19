__author__ = 'Aubrey'
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
from configs.base_configs import MethodConfigs

class TargetTranfer(method.Method):
    def __init__(self, configs=MethodConfigs()):
        super(TargetTranfer, self).__init__(configs)
        self.base_learner = method.SKLLogisticRegression(configs)
        self.cv_params = {}
        self.base_learner.experiment_results_class = self.experiment_results_class

    def train(self, data):
        self.base_learner.train_and_test(data)

    def train_and_test(self, data):
        data_copy = self._prepare_data(data,include_unlabeled=True)
        results = super(TargetTranfer, self).train_and_test(data_copy)
        return results

    def _prepare_data(self, data, include_unlabeled=True):
        target_labels = self.configs.target_labels
        data_copy = data.get_transfer_subset(target_labels,include_unlabeled=include_unlabeled)

        data_copy = data_copy.get_subset(data_copy.is_target)

        #TODO: Not sure why I was doing this before
        #is_source = ~data_copy.has_true_label(target_labels)
        #data_copy.type[is_source] = data_lib.TYPE_SOURCE
        #data_copy.is_train[is_source] = True
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
    def __init__(self, configs=MethodConfigs()):
        super(FuseTransfer, self).__init__(configs)
        self.use_oracle = False
        #self.target_weight_scale = None
        self.target_weight_scale = .75
        self.label_transform = NanLabelBinarizer()

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
        if data.data_set_ids is not None:
            #assert source_labels is None
            #assert target_labels is None
            #data_copy.type[data_copy.data_set_ids > 0] = data_lib.TYPE_SOURCE
            for i in source_labels:
                data_copy.type[data_copy.data_set_ids == i] = data_lib.TYPE_SOURCE
            return data_copy
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
        if getattr(self, 'use_all_source', False):
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

class StackingTransfer(FuseTransfer):
    def __init__(self, configs=MethodConfigs()):
        super(StackingTransfer, self).__init__(configs)
        #from far_transfer_methods import GraphTransferNW
        self.base_learner = method.SKLRidgeRegression(deepcopy(configs))
        self.source_learner = method.NadarayaWatsonMethod(deepcopy(configs))
        self.target_learner = method.NadarayaWatsonMethod(deepcopy(configs))
        self.joint_cv = getattr(configs, 'joint_cv', False)
        self.only_use_source_prediction = False
        self.use_all_source = True
        self.source_only = False
        self.target_only = False
        self.just_bias = False
        self.linear_source = False
        if self.target_only or self.source_only or self.linear_source:
            self.joint_cv = False
        if self.just_bias:
            self.base_learner.cv_params = {
                'alpha': [1e16]
            }
            self.joint_cv = False
        if self.linear_source:
            self.target_learner.cv_params = {
                'sigma': [1]
            }
        if self.joint_cv:
            self.cv_params = self.base_learner.cv_params.copy()
            self.cv_params.update(self.target_learner.cv_params)
            self.base_learner.cv_params = None
            self.target_learner.cv_params = None

        sub_configs = deepcopy(configs)

        #self.source_learner = method.NadarayaWatsonKNNMethod(deepcopy(sub_configs))
        #self.target_learner = method.NadarayaWatsonKNNMethod(deepcopy(sub_configs))
        self.use_validation = configs.use_validation


    def _switch_labels(self, x, old, new):
        x_new = deepcopy(x)
        for o, n in zip(old[:], new[:]):
            x_new[x == o] = n
        return x_new

    def _get_stacked_data(self, data):
        y_source = np.expand_dims(self.source_learner.predict(data).y, 1)
        y_target = np.expand_dims(self.target_learner.predict(data).y, 1)
        if not data.is_regression:
            classes = data.classes
            new_classes = np.asarray([0,1])
            y_source = self._switch_labels(y_source, classes, new_classes)
            y_target = self._switch_labels(y_target, classes, new_classes)
        if self.linear_source:
            y_target[:] = 0
        x = np.hstack((y_target, y_source))
        data_stacked = deepcopy(data)
        data_stacked.x = x
        return data_stacked

    def train(self, data):
        if data.n_train_labeled == 0 or self.source_only:
            self.only_use_source_prediction = True
            return
        if self.joint_cv:
            self.target_learner.set_params(sigma=self.sigma)
            self.base_learner.set_params(alpha=self.alpha)
            self.target_learner.train(data)
        else:
            self.target_learner.train_and_test(data)
        self.only_use_source_prediction = False
        if self.target_only:
            return
        #Need unlabeled data if using validation data for parameter tuning
        #I = data.is_labeled & data.is_target
        I = data.is_target
        stacked_data = self._get_stacked_data(data).get_subset(I)
        if self.joint_cv:
            self.base_learner.train(stacked_data)
        else:
            self.base_learner.train_and_test(stacked_data)
        if not self.running_cv:
            '''
            dict = {
                'source': self.source_learner.sigma,
                'target': self.target_learner.sigma,
                'lambda': self.base_learner.alpha
            }
            '''
            print 'Stacking params: ' + str(dict)

    def predict(self, data):
        if self.only_use_source_prediction or self.source_only:
            return self.source_learner.predict(data)
        if self.target_only:
            return self.target_learner.predict(data)
        stacked_data = self._get_stacked_data(data)
        return self.base_learner.predict(stacked_data)

    def _prepare_data(self, data, include_unlabeled=True):
        data = super(StackingTransfer, self)._prepare_data(data, include_unlabeled)
        if not self.target_only:
            source_data = data.get_subset(data.is_source)
            if self.preprocessor is not None:
                source_data = self.preprocessor.preprocess(source_data, self.configs)
            self.source_learner.configs.source_labels = None
            self.source_learner.configs.target_labels = None
            source_data.set_target()
            self.source_learner.train_and_test(source_data)
        target_data = data.get_subset(data.is_target)
        return target_data

    @property
    def prefix(self):
        s = 'StackTransfer+' + self.base_learner.prefix
        if self.preprocessor is not None and self.preprocessor.prefix() is not None:
            s += '-' + self.preprocessor.prefix()
        if getattr(self, 'source_only', False):
            s += '-source'
        if getattr(self, 'target_only', False):
            s += '-target'
        if getattr(self, 'just_bias', False):
            s += '-bias'
        if getattr(self, 'linear_source', False):
            s += '-linearSource'
        if getattr(self, 'joint_cv', False):
            s += '-jointCV'
        if getattr(self, 'use_validation', False):
            s += '-VAL'
        return s


class HypothesisTransfer(FuseTransfer):

    WEIGHTS_ALL = 0
    WEIGHTS_JUST_TARGET = 1
    WEIGHTS_JUST_OPTIMAL = 2
    WEIGHTS_JUST_FIRST = 3
    def __init__(self, configs=MethodConfigs()):
        super(HypothesisTransfer, self).__init__(configs)
        self.cv_params = {
            'C': self.create_cv_params(-5,5),
            'C2': self.create_cv_params(-5, 5),
            'C3': self.create_cv_params(-5, 5),
        }
        self.w = None
        self.b = None

        #self.base_source_learner = method.SKLRidgeClassification(deepcopy(configs))
        self.base_source_learner = None
        self.label_transform = None

        self.source_w = []
        self.transform = StandardScaler()
        #self.transform = None
        self.use_oracle = False
        self.tune_C = False
        #self.weight_type = HypothesisTransfer.WEIGHTS_ALL
        #self.weight_type = HypothesisTransfer.WEIGHTS_JUST_TARGET
        #self.weight_type = HypothesisTransfer.WEIGHTS_JUST_OPTIMAL
        self.weight_type = HypothesisTransfer.WEIGHTS_JUST_FIRST
        if hasattr(configs, 'weight_type'):
            self.weight_type = configs.weight_type
        self.oracle_data_set_ids = configs.oracle_data_set_ids
        self.c_value = None
        self.use_test_error_for_model_selection = configs.use_test_error_for_model_selection
        if self.weight_type == HypothesisTransfer.WEIGHTS_JUST_TARGET:
            del self.cv_params['C2']
            del self.cv_params['C3']
            self.C2 = 0
            self.C3 = 0
        elif not getattr(self, 'tune_C', True):
            del self.cv_params['C']
            self.C = 0


    def train_and_test(self, data):
        #data = data.get_subset(data.data_set_ids == 0)
        source_labels = self.configs.source_labels
        data = self._prepare_data(data)
        target_data = data.get_subset(data.data_set_ids == 0)
        #self.cv_params['C'] = np.zeros(1)
        if self.weight_type != HypothesisTransfer.WEIGHTS_JUST_TARGET:
            base_configs = deepcopy(self.configs)
            base_configs.weight_type = HypothesisTransfer.WEIGHTS_JUST_TARGET
            self.base_source_learner = HypothesisTransfer(base_configs)
            self.base_source_learner.cv_use_data_type = False
            self.base_source_learner.use_test_error_for_model_selection = False
            #self.base_source_learner.cv_params['C'] = np.zeros(1)

            #for i, s in enumerate(source_labels):
            for data_set_id in np.unique(data.data_set_ids):
                if data_set_id == 0:
                    continue
                #source_inds = data.get_transfer_inds(s)
                source_data = data.get_subset(data.data_set_ids == data_set_id)
                source_data.data_set_ids[:] = 0
                source_data.is_target[:] = True
                self.base_source_learner.train_and_test(source_data)
                best_params = self.base_source_learner.best_params
                w = np.squeeze(self.base_source_learner.w)
                w /= np.linalg.norm(w)
                b = self.base_source_learner.b
                self.source_w.append(w)
                pass
            ws1 = self.source_w[0]
            ws2 = self.source_w[1]
            target_data_copy = deepcopy(target_data)
            target_data_copy.is_train[:] = True
            target_data_copy.y = target_data_copy.true_y
            self.base_source_learner.train_and_test(target_data_copy)
            wt = np.squeeze(self.base_source_learner.w)
            wt /= np.linalg.norm(wt)
            d1 = norm(ws1-wt)
            d2 = norm(ws2-wt)
            pass

        o = super(HypothesisTransfer, self).train_and_test(target_data)
        print 'c: ' + str(np.squeeze(self.c_value))
        return o

    def estimate_c(self, data):
        x = data.x[data.is_labeled & data.is_train]
        if self.transform is not None:
            x = self.transform.fit_transform(x)
        y = data.y[data.is_labeled]
        if self.label_transform is not None:
            y = self.label_transform.fit_transform(y)
        n = y.size
        p = data.p
        c = cvx.Variable(len(self.source_w))
        #ws1 = self.source_w[0]
        #ws2 = self.source_w[1]

        ws = 0
        for i, wsi in enumerate(self.source_w):
            ws += wsi * c[i]

        constraints = [c >= 0]
        constraint_methods = {
            HypothesisTransfer.WEIGHTS_JUST_OPTIMAL,
            HypothesisTransfer.WEIGHTS_JUST_FIRST
        }
        found_first = False
        if self.weight_type in constraint_methods:
            for i in range(c.size[0]):
                id = i + 1
                is_oracle = id in self.oracle_data_set_ids
                just_first = self.weight_type == HypothesisTransfer.WEIGHTS_JUST_FIRST and found_first
                if is_oracle and not just_first:
                    found_first = True
                    continue
                constraints.append(c[i] == 0)
        loss = 0
        for i in range(y.size):
            xi = x[i, :]
            yi = y[i]
            x_mi = np.delete(x, i, axis=0)
            y_mi = np.delete(y, i, axis=0)
            b_mi = y_mi.mean()
            A = x_mi.T.dot(x_mi) + (self.C + self.C2) * np.eye(p)
            k = x_mi.T.dot(y_mi) - x_mi.T.sum(1) * b_mi + self.C2 * ws
            w_mi = scipy.linalg.inv(A) * k
            loss += cvx.power(w_mi.T * xi + b_mi - yi, 2)
        reg = cvx.norm2(c)**2
        #reg = cvx.norm2(c)
        obj = cvx.Minimize(loss + self.C3 * reg)
        prob = cvx.Problem(obj, constraints)
        assert prob.is_dcp()
        try:
            prob.solve(cvx.SCS)
            c_value = np.asarray(c.value)
        except Exception as e:
            print str(e)
            c_value = np.zeros(p)
        # c_value[np.abs(c_value) <= 1e-4] = 0
        # assert np.all(c_value >= 0)
        c_value[c_value < 0] = 0
        return c_value

    def train(self, data):
        x = data.x[data.is_labeled & data.is_train]
        if self.transform is not None:
            x = self.transform.fit_transform(x)
        y = data.y[data.is_labeled]
        if self.label_transform is not None:
            y = self.label_transform.fit_transform(y)
        n = y.size
        p = data.p
        self.b = y.mean()

        #print str(np.squeeze(c_value))
        if self.weight_type == HypothesisTransfer.WEIGHTS_JUST_TARGET:
            c_value = np.zeros(len(self.source_w))
            #ws1 = 0
            #ws2 = 0
        else:
            c_value = self.estimate_c(data)
            #ws1 = self.source_w[0]
            #ws2 = self.source_w[1]
        ws = 0
        for i, wsi in enumerate(self.source_w):
            ws += wsi*c_value[i]
        A = x.T.dot(x) + (self.C + self.C2)*np.eye(p)
        #k = x.T.dot(y) - x.T.sum(1)*self.b + self.C2*(ws1*c_value[0] + ws2*c_value[1])
        k = x.T.dot(y) - x.T.sum(1) * self.b + self.C2 * ws
        self.w = np.linalg.solve(A, k)
        self.c_value = c_value
        pass

    def predict(self, data):
        o = Output(data)
        x = data.x
        if self.transform is not None:
            x = self.transform.transform(x)
        y = x.dot(self.w) + self.b
        #y = np.round(y)
        #y[y >= .5] = 1
        #y[y < .5] = 0
        y = np.sign(y)
        o.y = y
        o.fu = y
        if self.label_transform is not None:
            o.true_y = self.label_transform.transform(o.true_y)

        if not self.running_cv:
            is_correct = (o.y == o.true_y)
            mean_train = is_correct[o.is_train].mean()
            mean_test = is_correct[o.is_test].mean()
            mean_train_labeled = is_correct[data.is_train & data.is_labeled].mean()
            pass
        return o

    @property
    def prefix(self):
        s = 'HypTransfer'
        weight_type = getattr(self, 'weight_type', HypothesisTransfer.WEIGHTS_ALL)
        if weight_type == HypothesisTransfer.WEIGHTS_JUST_TARGET:
            s += '-target'
        else:
            if weight_type == HypothesisTransfer.WEIGHTS_JUST_OPTIMAL:
                s += '-optimal'
            elif weight_type == HypothesisTransfer.WEIGHTS_JUST_FIRST:
                s += '-first'
            if not getattr(self, 'tune_C', False):
                s += '-noC'
        if getattr(self, 'use_test_error_for_model_selection', False):
            s += '-TEST'
        return s


class ModelSelectionTransfer(method.ModelSelectionMethod):
    def __init__(self, configs=MethodConfigs()):
        super(ModelSelectionTransfer, self).__init__(configs)
        self.methods.append(TargetTranfer(configs))
        self.methods.append(FuseTransfer(configs))
        for m in self.methods:
            m.base_learner = method.NadarayaWatsonMethod(configs)

    @property
    def prefix(self):
        return 'ModelSelTransfer'

class ReweightedTransfer(method.Method):
    def __init__(self, configs=MethodConfigs()):
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

