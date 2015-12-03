__author__ = 'Aubrey'
import method
import copy
from data import data as data_lib
from results_class import results as results_lib
from utility import array_functions


class TargetTranfer(method.Method):
    def __init__(self, configs=None):
        super(TargetTranfer, self).__init__(configs)
        self.base_learner = method.SKLLogisticRegression(configs)
        self.cv_params = {}
        self.base_learner.experiment_results_class = self.experiment_results_class

    def train(self, data):
        self.base_learner.train_and_test(data)

    def train_and_test(self, data):
        data_copy = self._prepare_data(data)
        return super(TargetTranfer, self).train_and_test(data_copy)

    def _prepare_data(self, data):
        target_labels = self.configs.target_labels
        data_copy = data.get_with_labels(target_labels)
        return data_copy

    def predict(self, data):
        return self.base_learner.predict(data)

    @property
    def prefix(self):
        return 'TargetTransfer+' + self.base_learner.prefix

class FuseTransfer(TargetTranfer):
    def __init__(self, configs=None):
        super(FuseTransfer, self).__init__(configs)

    def _prepare_data(self, data):
        source_labels = self.configs.source_labels
        target_labels = self.configs.target_labels
        data_copy = copy.deepcopy(data)
        source_inds = array_functions.find_set(data_copy.true_y,source_labels)
        data_copy.change_labels(source_labels,target_labels)
        data_copy.type[source_inds] = data_lib.TYPE_SOURCE
        data_copy = data_copy.get_with_labels(target_labels)
        data_copy.is_train[data_copy.is_source] = True
        return data_copy

    @property
    def prefix(self):
        return 'FuseTransfer+' + self.base_learner.prefix




