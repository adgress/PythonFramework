import methods.local_transfer_methods

__author__ = 'Aubrey'
from collections import OrderedDict
from configs import base_configs as bc
import numpy as np
from data_sets import create_data_set
from loss_functions import loss_function
from data_sets import create_data_set
from utility import array_functions
from utility import helper_functions
from results_class import results as results_lib


def create_project_configs():
    return ProjectConfigs()

pc_fields_to_copy = bc.pc_fields_to_copy + [
]
#data_set_to_use = bc.DATA_BOSTON_HOUSING
#data_set_to_use = bc.DATA_SYNTHETIC_LINEAR_REGRESSION
data_set_to_use = bc.DATA_ADIENCE_ALIGNED_CNN_1

data_sets_for_exps = [data_set_to_use]

active_iterations = 2
active_items_per_iteration = 50
use_pairwise = False
num_pairwise = 10

run_active_experiments = False

run_experiments = True
show_legend_on_all = True

max_rows = 3

synthetic_dim = 1
if helper_functions.is_laptop():
    use_pool = False
    pool_size = 4
else:
    use_pool = True
    pool_size = 24

class ProjectConfigs(bc.ProjectConfigs):
    def __init__(self, data_set=None):
        super(ProjectConfigs, self).__init__()
        self.project_dir = 'active'
        self.use_pool = use_pool
        self.pool_size = pool_size
        if run_active_experiments:
            self.method_results_class = results_lib.ActiveMethodResults
        if data_set is None:
            data_set = data_set_to_use
        self.set_data_set(data_set)
        self.num_splits = 30

    def set_data_set(self, data_set):
        self.data_set = data_set
        if data_set == bc.DATA_BOSTON_HOUSING:
            self.set_boston_housing()
            self.num_labels = [5, 10, 20]
            if run_active_experiments:
                self.num_labels = [5]
        elif data_set == bc.DATA_SYNTHETIC_LINEAR_REGRESSION:
            self.set_synthetic_linear_reg()
            self.num_labels = [10, 20, 40]
            if run_active_experiments:
                self.num_labels = [20]
        elif data_set == bc.DATA_ADIENCE_ALIGNED_CNN_1:
            self.set_adience_aligned_cnn_1()
            self.num_labels = [10, 20, 40]
            if run_active_experiments:
                self.num_labels = [20]


    def set_boston_housing(self):
        self.loss_function = loss_function.MeanSquaredError()
        self.cv_loss_function = loss_function.MeanSquaredError()
        self.data_dir = 'data_sets/boston_housing'
        self.data_name = 'boston_housing'
        self.results_dir = 'boston_housing'
        self.data_set_file_name = 'split_data.pkl'

    def set_synthetic_linear_reg(self):
        self.loss_function = loss_function.MeanSquaredError()
        self.cv_loss_function = loss_function.MeanSquaredError()
        self.data_dir = 'data_sets/synthetic_linear_reg500-50-1'
        self.data_name = 'synthetic_linear_reg500-50-1'
        self.results_dir = 'synthetic_linear_reg500-50-1'
        self.data_set_file_name = 'split_data.pkl'

    def set_adience_aligned_cnn_1(self):
        self.loss_function = loss_function.MeanSquaredError()
        self.cv_loss_function = loss_function.MeanSquaredError()
        self.data_dir = 'data_sets/adience_aligned_cnn_1_per_instance_id'
        self.data_name = 'adience_aligned_cnn_1_per_instance_id'
        self.results_dir = 'adience_aligned_cnn_1_per_instance_id'
        self.data_set_file_name = 'split_data.pkl'


class MainConfigs(bc.MainConfigs):
    def __init__(self, pc):
        super(MainConfigs, self).__init__()
        #pc = create_project_configs()
        self.copy_fields(pc,pc_fields_to_copy)
        from methods import method
        from methods import active_methods
        method_configs = MethodConfigs(pc)
        method_configs.active_iterations = active_iterations
        method_configs.active_items_per_iteration = active_items_per_iteration
        method_configs.metric = 'euclidean'
        method_configs.use_pairwise = use_pairwise
        method_configs.num_pairwise = num_pairwise

        #active = active_methods.ActiveMethod(method_configs)
        active = active_methods.RelativeActiveMethod(method_configs)
        active.base_learner = method.RelativeRegressionMethod(method_configs)
        relative_reg = method.RelativeRegressionMethod(method_configs)
        ridge_reg = method.SKLRidgeRegression(method_configs)
        if run_active_experiments:
            self.learner = active
        else:
            self.learner = relative_reg
            #self.learner = ridge_reg

class MethodConfigs(bc.MethodConfigs):
    def __init__(self, pc):
        super(MethodConfigs, self).__init__()
        self.copy_fields(pc,pc_fields_to_copy)

class VisualizationConfigs(bc.VisualizationConfigs):
    def __init__(self, data_set=None):
        super(VisualizationConfigs, self).__init__()
        pc = ProjectConfigs(data_set)
        self.copy_fields(pc,pc_fields_to_copy)
        self.files = OrderedDict()
        if run_active_experiments:
            self.files['RelActiveRandom+SKL-RidgeReg.pkl'] = 'Random Pairwise, SKLRidge'
            self.files['ActiveRandom+SKL-RidgeReg.pkl'] = 'Random, SKLRidge'
            self.files['RelActiveRandom+RelReg-cvx-log-with-log-noLinear-TEST.pkl'] = 'TEST: RandomPairwise, RelReg'
        else:
            self.files['RelReg-noPairwiseReg.pkl'] = 'Relative Ridge no Pairwise Reg'
            self.files['RelReg-cvx-log-numRandPairs=50.pkl'] = 'Relative Ridge Log, 50 random pairs'

            #self.files['RelReg-cvx-log-with-log-scale-numRandPairs=10-noLinear-TEST.pkl'] = 'Test: Relative Ridge Log with Log-scale, no Linear , 10 random pairs'
            self.files['RelReg-cvx-log-with-log-scale-numRandPairs=50-noLinear.pkl'] = 'Relative Ridge Log with Log-scale, no Linear , 50 random pairs'
            if pc.data_set == bc.DATA_ADIENCE_ALIGNED_CNN_1:
                self.files['SKL-RidgeReg.pkl'] = 'SKL RIdge'

        self.figsize = (7,7)
        self.borders = (.1,.9,.9,.1)
        self.data_set_to_use = pc.data_set
        self.title = bc.data_name_dict.get(self.data_set_to_use, 'Unknown Data Set')
        self.show_legend_on_all = show_legend_on_all
        self.x_axis_string = 'Number of labeled instances'


class BatchConfigs(bc.BatchConfigs):
    def __init__(self, pc):
        super(BatchConfigs, self).__init__()
        self.config_list = [MainConfigs(pc)]
        from experiment.experiment_manager import MethodExperimentManager
        self.method_experiment_manager_class = MethodExperimentManager