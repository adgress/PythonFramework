__author__ = 'Aubrey'
from collections import OrderedDict
from configs import base_configs as bc
from loss_functions import loss_function
from utility import helper_functions
from results_class import results as results_lib
from sklearn import grid_search
from utility import helper_functions
from methods import mixed_feature_guidance
from copy import deepcopy
import numpy as np

# Command line arguments for ProjectConfigs
arguments = None

def create_project_configs():
    return ProjectConfigs()

pc_fields_to_copy = bc.pc_fields_to_copy + [
    'subset_size',
    'num_samples'
]

classification_data_sets = {
    bc.DATA_MNIST
}

#data_set_to_use = bc.DATA_MNIST
#data_set_to_use = bc.DATA_BOSTON_HOUSING
#data_set_to_use = bc.DATA_CONCRETE
data_set_to_use = bc.DATA_CLIMATE_MONTH

METHOD_CLUSTER = 0
METHOD_CLUSTER_SPLIT = 1
METHOD_CLUSTER_GRAPH = 2
METHOD_CLUSTER_SUBMODULAR = 3

LOSS_Y = 0
LOSS_P = 1
LOSS_NOISY = 2
LOSS_ENTROPY = 3

#instance_selection_method = METHOD_CLUSTER
instance_selection_method = METHOD_CLUSTER_SUBMODULAR

loss_to_use = LOSS_Y

run_experiments = True
run_batch_experiments = True
use_training = True
all_data_sets = [data_set_to_use]


max_rows = 1

other_pc_configs = {
}

other_method_configs = {
    'include_size_in_file_name': False,
}

if helper_functions.is_laptop():
    use_pool = False
    pool_size = 1
else:
    use_pool = False
    pool_size = 1

arguments = None
def apply_arguments(configs):
    if arguments.num_labels is not None:
        configs.overwrite_num_labels = arguments.num_labels
    if arguments.split_idx is not None:
        configs.split_idx = arguments.split_idx


class ProjectConfigs(bc.ProjectConfigs):
    def __init__(self, data_set=None, use_arguments=True):
        super(ProjectConfigs, self).__init__()
        self.project_dir = 'instance_selection'
        self.use_pool = use_pool
        self.pool_size = pool_size
        if data_set is None:
            data_set = data_set_to_use
        self.set_data_set(data_set)
        self.num_splits = 30
        self.subset_size = 10
        self.num_samples = 10
        self.instance_selection_method = instance_selection_method
        self.loss_to_use = loss_to_use
        if use_arguments and arguments is not None:
            apply_arguments(self)

        for key, value in other_method_configs.items():
            setattr(self, key, value)


    def set_data_set(self, data_set):
        self.data_set = data_set

        if data_set == bc.DATA_MNIST:
            self.set_data_set_defaults('mnist')
            self.num_labels = [50]
            self.subset_size = 10
            self.num_samples = 10
        elif data_set == bc.DATA_BOSTON_HOUSING:
            self.set_data_set_defaults('boston_housing')
            self.num_labels = [200]
            self.subset_size = 10
            self.num_samples = 10
        elif data_set == bc.DATA_CONCRETE:
            self.set_data_set_defaults('concrete')
            self.num_labels = [200]
            self.subset_size = 10
            self.num_samples = 10
        elif data_set == bc.DATA_CLIMATE_MONTH:
            self.set_data_set_defaults('climate-month')
            self.num_labels = [500]
            self.subset_size = 10
            self.num_samples = 10
        else:
            assert False
        '''
        if self.include_size_in_file_name:
            assert len(self.num_labels) == 1
        '''




class MainConfigs(bc.MainConfigs):
    def __init__(self, pc):
        super(MainConfigs, self).__init__()
        #pc = create_project_configs()
        self.copy_fields(pc,pc_fields_to_copy)
        from methods import method
        from methods import preprocessing
        from methods import instance_selection
        from methods import wrapper_methods
        from methods.wrapper_methods import *
        from sklearn.preprocessing import StandardScaler
        method_configs = MethodConfigs(pc)
        for key in other_method_configs.keys():
            setattr(method_configs, key, getattr(pc,key))
        method_configs.results_features = ['y', 'true_y']
        method_configs.instance_subset = 'is_test'

        pipeline = PipelineMethod(method_configs)
        max_std = None
        if pc.data_set == bc.DATA_MNIST:
            pipeline.preprocessing_pipeline = [
                PipelineChangeClasses({
                    0: 4,
                    8: 7
                }),
                PipelineSelectClasses([4, 7]),
                PipelineSelectLabeled(),
                PipelineMakeRegression(),
                PipelineSKLTransform(PCA(n_components=2)),
                PipelineAddClusterNoise(
                    num_clusters=0,
                    n_per_cluster=15
                )
            ]
        elif pc.data_set == bc.DATA_CLIMATE_MONTH:
            pipeline.preprocessing_pipeline = [
                PipelineSelectLabeled(),
                PipelineSelectDataIDs(ids=[0]),
                PipelineAddClusterNoise(
                    num_clusters=0,
                    n_per_cluster=5,
                    flip_labels=False,
                    y_offset=10,
                    save_y_orig=True
                )
            ]
            max_std = .6
        else:
            pipeline.preprocessing_pipeline = [
                PipelineSelectLabeled(),
                PipelineSKLTransform(StandardScaler()),
                PipelineSKLTransform(PCA(n_components=2)),
                PipelineAddClusterNoise(
                    num_clusters=0,
                    n_per_cluster=15,
                    flip_labels=False,
                    y_offset=5,
                    save_y_orig=True
                )
            ]
        if pc.instance_selection_method == METHOD_CLUSTER:
            sisc = instance_selection.SupervisedInstanceSelectionCluster(method_configs)
            sisc.subset_size = 8
            sisc.num_samples = 8
        elif pc.instance_selection_method == METHOD_CLUSTER_GRAPH:
            sisc = instance_selection.SupervisedInstanceSelectionClusterGraph(method_configs)
            method_configs.results_features = ['y', 'true_y']
            sisc.configs.results_features = ['res_total', 'res_total']
            sisc.configs.cv_loss_function = loss_function.LossNorm()
            sisc.configs.use_training = use_training
            sisc.subset_size = 8
            sisc.num_samples = 8
        elif pc.instance_selection_method == METHOD_CLUSTER_SUBMODULAR:
            sisc = instance_selection.SupervisedInstanceSelectionSubmodular(method_configs)
            method_configs.results_features = ['y', 'true_y']
            sisc.configs.results_features = ['res_total', 'res_total']
            sisc.configs.cv_loss_function = loss_function.LossNorm()
            sisc.configs.use_training = use_training
            sisc.subset_size = 8
            sisc.num_samples = 8
        else:
            sisc = instance_selection.SupervisedInstanceSelectionClusterSplit(method_configs)
            sisc.subset_size = 5
            sisc.num_samples = 5
            if max_std is not None:
                sisc.max_std = max_std
        pipeline.base_learner = sisc
        self.learner = pipeline


class MethodConfigs(bc.MethodConfigs):
    def __init__(self, pc):
        super(MethodConfigs, self).__init__()
        self.copy_fields(pc,pc_fields_to_copy)

class BatchConfigs(bc.BatchConfigs):
    def __init__(self, pc):
        super(BatchConfigs, self).__init__()
        from experiment.experiment_manager import MethodExperimentManager
        self.method_experiment_manager_class = MethodExperimentManager
        self.config_list = [MainConfigs(pc)]
        if run_batch_experiments:
            self.config_list = []
            p = deepcopy(pc)
            p.instance_selection_method = METHOD_CLUSTER
            self.config_list.append(MainConfigs(p))

            p = deepcopy(pc)
            p.instance_selection_method = METHOD_CLUSTER_SPLIT
            self.config_list.append(MainConfigs(p))

            p = deepcopy(pc)
            p.instance_selection_method = METHOD_CLUSTER_GRAPH
            self.config_list.append(MainConfigs(p))

            p = deepcopy(pc)
            p.instance_selection_method = METHOD_CLUSTER_SUBMODULAR
            self.config_list.append(MainConfigs(p))


class VisualizationConfigs(bc.VisualizationConfigs):
    def __init__(self, data_set=None, **kwargs):
        super(VisualizationConfigs, self).__init__(data_set, **kwargs)
        self.max_rows = max_rows
        pc = ProjectConfigs(data_set)
        self.copy_fields(pc,pc_fields_to_copy)

        self.data_set_to_use = pc.data_set
        self.title = bc.data_name_dict.get(self.data_set_to_use, 'Unknown Data Set')
        self.show_legend_on_all = True
        self.x_axis_string = 'Number of labeled instances'
        self.ylims = None
        self.generate_file_names(pc)

        viz_loss_function = loss_function.MeanSquaredError()
        self.always_show_y_label = True
        is_regression = not self.data_set_to_use in classification_data_sets

        instance_subset = 'is_train'
        if not hasattr(self, 'loss_to_use'):
            self.loss_to_use = loss_to_use
        if self.loss_to_use == LOSS_Y:
            results_features = ['y', 'true_y']
            self.y_axis_string = 'Prediction Error'
        elif self.loss_to_use == LOSS_P:
            results_features = ['p', 'true_p']
            self.y_axis_string = 'P(X) Error'
        elif self.loss_to_use == LOSS_NOISY:
            results_features = ['is_noisy', 'is_selected']
            viz_loss_function = loss_function.LossAnyOverlap()
            self.y_axis_string = 'Noisy Error'
        elif self.loss_to_use == LOSS_ENTROPY:
            instance_subset = 'is_selected'
            results_features = ['y_orig', 'y_orig']
            viz_loss_function = loss_function.LossSelectedEntropy(is_regression=is_regression)
            self.y_axis_string = 'Selection Distribution Error'
        else:
            assert False


        self.instance_subset = instance_subset
        self.results_features = results_features
        self.loss_function = viz_loss_function

    def generate_file_names(self, pc):
        self.files = OrderedDict()
        self.files['Pipeline+SupervisedInstanceSelectionCluster.pkl'] = 'Cluster'
        self.files['Pipeline+SupervisedInstanceSelectionClusterSplit.pkl'] = 'Cluster Split'
        self.files['Pipeline+SupervisedInstanceSelectionClusterGraph.pkl'] = 'Cluster Graph'


'''
viz_params = [
    {'data_set': d} for d in all_data_sets
]
'''
viz_params = [
    {'loss_to_use': d} for d in [LOSS_Y, LOSS_P, LOSS_NOISY, LOSS_ENTROPY]
]