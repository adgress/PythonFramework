__author__ = 'Aubrey'
import copy

import numpy as np

import method
from data import data as data_lib


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
        #data_copy = data.get_with_labels(target_labels)
        return data_copy

    def predict(self, data):
        return self.base_learner.predict(data)

    @method.Method.estimated_error.getter
    def estimated_error(self):
        return self.base_learner.estimated_error

    @property
    def prefix(self):
        return 'TargetTransfer+' + self.base_learner.prefix

class FuseTransfer(TargetTranfer):
    def __init__(self, configs=None):
        super(FuseTransfer, self).__init__(configs)
        self.use_oracle = True
        self.target_weight_scale = None
        #self.target_weight_scale = .9

    def train(self, data):
        is_labeled_train = data.is_labeled & data.is_train
        n_labeled_target = (data.is_train & is_labeled_train).sum()
        n_labeled_source = (data.is_train & is_labeled_train).sum()
        data.instance_weights = np.ones(data.n)
        if self.target_weight_scale is not None:
            assert self.target_weight_scale > 0 and self.target_weight_scale <= 1
            data.instance_weights[data.is_source] /= (n_labeled_source)
            data.instance_weights[data.is_target] /= (n_labeled_target)
            data.instance_weights[data.is_target] *= self.target_weight_scale
            data.instance_weights[data.is_source] *= (1-self.target_weight_scale)
        super(FuseTransfer, self).train(data)

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
        source_inds = data_copy.get_transfer_inds(source_labels.ravel())
        if not data_copy.is_regression:
            data_copy.change_labels(source_labels,target_labels)
        data_copy.type[source_inds] = data_lib.TYPE_SOURCE
        data_copy = data_copy.get_transfer_subset(np.concatenate((source_labels.ravel(),target_labels)),include_unlabeled=include_unlabeled)
        data_copy.is_train[data_copy.is_source] = True
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
