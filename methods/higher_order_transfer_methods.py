from methods import method
from methods import transfer_methods
from methods import local_transfer_methods
from utility import array_functions
import copy
from copy import deepcopy
import numpy as np
from numpy.linalg import norm
from data import data as data_lib
from sklearn.neighbors import KernelDensity
from sklearn.grid_search import GridSearchCV
from results_class.results import Output
import cvxpy as cvx
import scipy
from data import data as data_lib

class DomainModelShiftMethod(transfer_methods.TargetTranfer):
    def __init__(self, configs=None):
        super(DomainModelShiftMethod, self).__init__(configs)
        self.base_learner = transfer_methods.StackingTransfer(configs)
        self.target_learner = method.NadarayaWatsonMethod(configs)

    def train(self, data):
        return
        #self.base_learner.train_and_test(data)

    def train_and_test(self, data):
        source_order = self.configs.source_domain_order
        target_order = self.configs.target_domain_order
        #results = super(DomainModelShiftMethod, self).train_and_test(data_copy)

        source_to_keep = array_functions.find_set(data.data_set_ids, source_order)
        source_data = data.get_subset(source_to_keep)
        source_data.y = source_data.true_y
        source_configs = deepcopy(self.configs)
        source_configs.labels_to_keep = source_order
        source_configs.labels_to_not_sample = np.asarray([source_order[0]])
        source_configs.source_labels = np.asarray([source_order[0]])
        source_configs.target_labels = np.asarray([source_order[1]])

        source_transformation = local_transfer_methods.OffsetTransfer(source_configs)
        source_transformation.use_validation = True
        source_transformation.train_and_test(source_data)

        target_to_keep = array_functions.find_set(data.data_set_ids, [target_order[0]])
        target_data = data.get_subset(target_to_keep)
        target_data.reveal_labels(target_data.data_set_ids == target_order[0])
        target_configs = deepcopy(self.configs)
        target_configs.labels_to_keep = np.asarray([target_order[0]])
        target_configs.source_labels = np.asarray([])
        target_configs.target_labels = np.asarray([target_order[0]])

        offset_labels = source_transformation.predict(target_data).y
        target_data.y = offset_labels
        target_data.true_y = offset_labels
        self.target_learner = method.NadarayaWatsonMethod(target_configs)
        self.target_learner.use_validation = True
        self.target_learner.train_and_test(target_data)

        t = data.get_subset(data.data_set_ids == target_order[1])
        return super(DomainModelShiftMethod, self).train_and_test(t)

    def predict(self, data):
        o = self.target_learner.predict(data)
        return o

    @property
    def prefix(self):
        return 'DomainModelShift-' + 'source=' + str(self.configs.source_domain_order) + '-target=' + str(self.configs.target_domain_order)