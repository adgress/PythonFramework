import methods.local_transfer_methods

__author__ = 'Aubrey'

from configs import base_configs as bc
import numpy as np
from data_sets import create_data_set
from loss_functions import loss_function
from data_sets import create_data_set
from utility import array_functions
from utility import helper_functions
from collections import OrderedDict
from copy import deepcopy
from methods import preprocessing
from methods import higher_order_transfer_methods
def create_project_configs():
    return ProjectConfigs()

pc_fields_to_copy = bc.pc_fields_to_copy + [
    'target_labels',
    'source_labels',
    'target_domain_order',
    'source_domain_order',
    'data_set'
]
data_set_to_use = bc.DATA_POLLUTION_2

show_legend_on_all = False
arguments = None
use_validation = False

run_experiments = True

all_data_sets = [data_set_to_use]


other_method_configs = {
}

def apply_arguments(configs):
    if arguments.num_labels is not None:
        configs.overwrite_num_labels = arguments.num_labels
    if arguments.split_idx is not None:
        configs.split_idx = arguments.split_idx

class ProjectConfigs(bc.ProjectConfigs):
    def __init__(self, data_set=None, use_arguments=True, **kwargs):
        super(ProjectConfigs, self).__init__()
        self.target_labels = np.empty(0)
        self.source_labels = np.empty(0)
        self.target_domain_order = None
        self.source_domain_order = None
        self.project_dir = 'higher_order_transfer'
        self.num_labels = range(40,201,40)
        if data_set is None:
            data_set = data_set_to_use
        for key, value in other_method_configs.items():
            setattr(self, key, value)
        for key, value in kwargs.iteritems():
            setattr(self, key, value)
        self.set_data_set(data_set)
        if use_arguments and arguments is not None:
            apply_arguments(self)

    def set_data_set(self, data_set):
        self.data_set = data_set

        if data_set == bc.DATA_POLLUTION_2:
            self.set_pollution()
            #self.num_labels = np.asarray([0, 5, 10, 20, 40])
            self.num_labels = np.asarray([0])
            self.target_labels = np.asarray([0])
            self.source_labels = np.asarray([1,2,3])
        assert self.source_labels.size > 0
        assert self.target_labels.size > 0
        self.labels_to_not_sample = self.source_labels.ravel()
        a = self.source_labels.ravel()
        self.labels_to_keep = np.concatenate((self.target_labels,a))

    def set_pollution(self):
        self.loss_function = loss_function.MeanSquaredError()
        self.cv_loss_function = loss_function.MeanSquaredError()
        s = 'pollution-[3 4]-500-norm'
        #s = 'pollution-[60 71]-500-norm'
        self.data_dir = 'data_sets/' + s
        self.data_name = s
        self.results_dir = s
        self.data_set_file_name = 'split_data.pkl'
        self.target_labels = np.asarray([0])
        self.source_labels = np.asarray([1, 2, 3])
        self.target_domain_order = np.asarray([1, 0])
        self.source_domain_order = np.asarray([3, 2])


class MainConfigs(bc.MainConfigs):
    def __init__(self, pc):
        super(MainConfigs, self).__init__()
        #pc = create_project_configs()
        self.copy_fields(pc,pc_fields_to_copy)
        from methods import transfer_methods
        from methods import method
        method_configs = MethodConfigs(pc)
        method_configs.metric = 'euclidean'
        method_configs.use_validation = use_validation


        target_nw = transfer_methods.TargetTranfer(method_configs)
        target_nw.base_learner = method.NadarayaWatsonMethod(method_configs)
        target_ridge = transfer_methods.TargetTranfer(method_configs)
        target_ridge.base_learner = method.SKLRidgeRegression(method_configs)


        stacked_nw1 = transfer_methods.StackingTransfer(method_configs)
        stacked_nw1.preprocessor = preprocessing.SelectSourcePreprocessor(sources_to_keep=[1])

        stacked_nw2 = transfer_methods.StackingTransfer(method_configs)
        stacked_nw2.preprocessor = preprocessing.SelectSourcePreprocessor(sources_to_keep=[2])

        stacked_nw3 = transfer_methods.StackingTransfer(method_configs)
        stacked_nw3.preprocessor = preprocessing.SelectSourcePreprocessor(sources_to_keep=[3])

        higher_order = higher_order_transfer_methods.DomainModelShiftMethod(method_configs)

        #self.learner = target_nw
        #self.learner = stacked_nw1
        #self.learner = stacked_nw2
        #self.learner = stacked_nw3
        self.learner = higher_order



class MethodConfigs(bc.MethodConfigs):
    def __init__(self, pc):
        super(MethodConfigs, self).__init__()
        self.copy_fields(pc,pc_fields_to_copy)
        self.target_labels = pc.target_labels
        self.source_labels = pc.source_labels


def append_suffix_to_files(dict, suffix, legend_suffix):
    d = OrderedDict()
    for key,value in dict.iteritems():
        f = helper_functions.remove_suffix(key, '.pkl')
        f += suffix + '.pkl'
        d[f] = value + legend_suffix
    return d



class VisualizationConfigs(bc.VisualizationConfigs):
    def __init__(self, data_set=None, **kwargs):
        super(VisualizationConfigs, self).__init__()
        pc = ProjectConfigs(data_set, **kwargs)
        self.copy_fields(pc,pc_fields_to_copy)
        self.max_rows = 2
        for key, value in kwargs.iteritems():
            setattr(self, key, value)
        self.files = OrderedDict()
        self.files['TargetTransfer+NW.pkl'] = 'Target Only'
        self.files['StackTransfer+SKL-RidgeReg-Sources=[1].pkl'] = 'Stacked: 1'
        self.files['StackTransfer+SKL-RidgeReg-Sources=[2].pkl'] = 'Stacked: 2'
        self.files['StackTransfer+SKL-RidgeReg-Sources=[3].pkl'] = 'Stacked: 3'
        self.files['DomainModelShift-source=[3 2]-target=[1 0].pkl'] = 'Domain Model Shift: 3-2 to 1-0'

        self.title = self.results_dir


class BatchConfigs(bc.BatchConfigs):
    def __init__(self, pc):
        super(BatchConfigs, self).__init__()
        self.config_list = []

        for d in all_data_sets:
            pc2 = ProjectConfigs(d)
            m = MainConfigs(pc2)
            self.config_list.append(m)

viz_params = [
    {'data_set': d} for d in all_data_sets
]