#import methods.constrained_methods
#import methods.local_transfer_methods
import methods.method

__author__ = 'Aubrey'
from collections import OrderedDict
from configs import base_configs as bc
from loss_functions import loss_function
from utility import helper_functions
from results_class import results as results_lib
from sklearn import grid_search
from copy import deepcopy
import numpy as np
# Command line arguments for ProjectConfigs
arguments = None


def create_project_configs():
    return ProjectConfigs()

pc_fields_to_copy = bc.pc_fields_to_copy + [
    'include_size_in_file_name',
    'active_method'
]
#data_set_to_use = bc.DATA_SYNTHETIC_LINEAR_REGRESSION
#data_set_to_use = bc.DATA_ADIENCE_ALIGNED_CNN_1

all_data_sets = [
    bc.DATA_SYNTHETIC_PIECEWISE,
    bc.DATA_BOSTON_HOUSING,
    #bc.DATA_MNIST,
    bc.DATA_CONCRETE,
    bc.DATA_WINE,
    bc.DATA_ZILLOW,
    bc.DATA_CLIMATE_MONTH,
    bc.DATA_IRS,
    bc.DATA_KC_HOUSING,
    bc.DATA_TAXI
]

data_set_to_use = bc.DATA_SYNTHETIC_PIECEWISE
#data_set_to_use = bc.DATA_TAXI
#data_set_to_use = bc.DATA_MNIST
#data_set_to_use = bc.DATA_SYNTHETIC_LINEAR_REGRESSION_10
#data_set_to_use = bc.DATA_BOSTON_HOUSING
#data_set_to_use = bc.DATA_CONCRETE
#data_set_to_use = bc.DATA_DROSOPHILIA
#data_set_to_use = bc.DATA_DS2


ACTIVE_RANDOM = 0
ACTIVE_CLUSTER = 1
ACTIVE_CLUSTER_PURITY = 2

active_method = ACTIVE_CLUSTER_PURITY

num_starting_labels = 0
active_iterations = 2
active_items_per_iteration = 10
cluster_scale = 1
max_target_items_for_large_data_sets = 300
large_data_sets = {
    bc.DATA_ZILLOW,
    bc.DATA_CLIMATE_MONTH,
    bc.DATA_IRS,
    bc.DATA_KC_HOUSING,
    bc.DATA_TAXI
}

fixed_sigma_x = False

# f_x here means f(x)
no_f_x = False
no_spectral_kernel = False
use_greedy_instance_selection = True
cluster_select_singleton = True
transfer_hyperparameters = False
use_knn = True
use_l1_loss = True

viz_for_paper = True

run_experiments = True

run_batchs_datasets = not run_experiments or True
run_batch_methods = True

use_test_error_for_model_selection = False


include_size_in_file_name = False

data_sets_for_exps = [data_set_to_use]
if run_batchs_datasets:
    data_sets_for_exps = all_data_sets

other_method_configs = {
    'scipy_opt_method': 'L-BFGS-B',
    'num_cv_splits': 10,
    'use_perfect_feature_selection': False,
    'use_test_error_for_model_selection': False,
    'use_validation': True,
    'use_uncertainty': False,
    'use_oed': False,
    'use_true_y': False,
    'num_features': None
}

run_batch = True
if helper_functions.is_laptop():
    run_batch = True

show_legend_on_all = False

max_rows = 1

if helper_functions.is_laptop():
    use_pool = False
    pool_size = 1
else:
    use_pool = False
    pool_size = 1

def apply_arguments(configs):
    if arguments.num_labels is not None:
        configs.overwrite_num_labels = arguments.num_labels
    if arguments.split_idx is not None:
        configs.split_idx = arguments.split_idx

class ProjectConfigs(bc.ProjectConfigs):
    def __init__(self, data_set=None, use_arguments=True):
        super(ProjectConfigs, self).__init__()
        self.project_dir = 'active_transfer'
        self.use_pool = use_pool
        self.pool_size = pool_size
        self.method_results_class = results_lib.ActiveMethodResults
        if data_set is None:
            data_set = data_set_to_use
        self.set_data_set(data_set)
        self.num_splits = 30
        self.active_method = active_method
        self.fixed_sigma_x = fixed_sigma_x
        self.no_spectral_kernel = no_spectral_kernel
        self.no_f_x = no_f_x
        self.use_greedy_instance_selection = use_greedy_instance_selection
        self.active_items_per_iteration = active_items_per_iteration
        self.cluster_select_singleton = cluster_select_singleton
        self.transfer_hyperparameters = transfer_hyperparameters
        self.use_knn = use_knn
        self.use_l1_loss = use_l1_loss
        if use_arguments and arguments is not None:
            apply_arguments(self)

        for key, value in other_method_configs.items():
            setattr(self, key, value)

        self.use_test_error_for_model_selection = use_test_error_for_model_selection

        self.include_size_in_file_name = include_size_in_file_name

    def set_data_set(self, data_set):
        self.data_set = data_set


        if data_set == bc.DATA_SYNTHETIC_PIECEWISE:
            self.set_data_set_defaults('synthetic_piecewise')
            self.num_labels = np.asarray([num_starting_labels])
            self.target_labels = np.zeros([1])
            self.source_labels = np.ones([1])
        elif data_set == bc.DATA_MNIST:
            self.set_data_set_defaults('mnist')
            self.num_labels = [num_starting_labels/2]
            self.target_labels = np.asarray([1, 3])
            self.source_labels = np.asarray([7, 8])
            self.loss_function = loss_function.ZeroOneError()
            self.cv_loss_function = loss_function.ZeroOneError()
        elif data_set == bc.DATA_BOSTON_HOUSING:
            self.set_data_set_defaults('boston_housing-13(transfer)')
            self.num_labels = [num_starting_labels]
            self.target_labels = np.asarray([0])
            self.source_labels = np.asarray([1])
        elif data_set == bc.DATA_WINE:
            self.set_data_set_defaults('wine-small-11')
            self.num_labels = [num_starting_labels]
            self.target_labels = np.asarray([0])
            self.source_labels = np.asarray([1])
        elif data_set == bc.DATA_CONCRETE:
            self.set_data_set_defaults('concrete-7')
            self.num_labels = [num_starting_labels]
            self.target_labels = np.asarray([1])
            self.source_labels = np.asarray([3])
        elif data_set == bc.DATA_CLIMATE_MONTH:
            self.set_data_set_defaults('climate-month', source_labels=[0], target_labels=[4], is_regression=True)
            self.num_labels = np.asarray([num_starting_labels])
        elif data_set == bc.DATA_IRS:
            self.set_data_set_defaults('irs-income', source_labels=[0], target_labels=[1], is_regression=True)
            self.num_labels = np.asarray([num_starting_labels])
        elif data_set == bc.DATA_KC_HOUSING:
            self.set_data_set_defaults('kc-housing-spatial-floors', source_labels=[0], target_labels=[1], is_regression=True)
            self.num_labels = np.asarray([num_starting_labels])
        elif data_set == bc.DATA_ZILLOW:
            self.set_data_set_defaults('zillow-traffic', source_labels=[1], target_labels=[0], is_regression=True)
            #self.set_data_set_defaults('zillow', source_labels=[1], target_labels=[0], is_regression=True)
            self.num_labels = np.asarray([num_starting_labels])
        elif data_set == bc.DATA_TAXI:
            #self.set_data_set_defaults('taxi2-20', source_labels=[1], target_labels=[0], is_regression=True)
            #self.set_data_set_defaults('taxi2-50', source_labels=[1], target_labels=[0], is_regression=True)
            #self.set_data_set_defaults('taxi2', source_labels=[0], target_labels=[1], is_regression=True)
            #self.set_data_set_defaults('taxi3', source_labels=[1], target_labels=[0], is_regression=True)
            self.set_data_set_defaults('taxi', source_labels=[1], target_labels=[0], is_regression=True)
            #self.num_labels = np.asarray([5, 10, 20, 40, 100, 200, 400, 800])
            self.num_labels = np.asarray([num_starting_labels])
        else:
            assert False, 'unknown transfer data set'

        assert self.source_labels.size > 0
        assert self.target_labels.size > 0
        self.labels_to_not_sample = self.source_labels.ravel()
        a = self.source_labels.ravel()
        self.labels_to_keep = np.concatenate((self.target_labels, a))


class MainConfigs(bc.MainConfigs):
    def __init__(self, pc):
        super(MainConfigs, self).__init__()
        #pc = create_project_configs()
        self.copy_fields(pc,pc_fields_to_copy)
        from methods import method
        from methods import active_methods
        from methods import wrapper_methods
        from methods import semisupervised
        method_configs = MethodConfigs(pc)
        method_configs.cluster_select_singleton = cluster_select_singleton
        method_configs.active_iterations = active_iterations
        method_configs.active_items_per_iteration = pc.active_items_per_iteration
        method_configs.metric = 'euclidean'
        method_configs.num_starting_labels = num_starting_labels
        method_configs.fixed_sigma_x = pc.fixed_sigma_x
        method_configs.no_f_x = pc.no_f_x
        method_configs.no_spectral_kernel = pc.no_spectral_kernel
        method_configs.transfer_hyperparameters = pc.transfer_hyperparameters
        method_configs.use_greedy_instance_selection = pc.use_greedy_instance_selection
        method_configs.use_knn = pc.use_knn
        method_configs.use_l1_loss = pc.use_l1_loss

        for key in other_method_configs.keys():
            setattr(method_configs, key, getattr(pc,key))

        method_configs.use_test_error_for_model_selection = pc.use_test_error_for_model_selection

        if pc.active_method == ACTIVE_RANDOM:
            active = active_methods.ActiveMethod(method_configs)
        elif pc.active_method == ACTIVE_CLUSTER:
            active = active_methods.ClusterActiveMethod(method_configs)
            active.cluster_scale = cluster_scale
        elif pc.active_method == ACTIVE_CLUSTER_PURITY:
            active = active_methods.ClusterPurityActiveMethod(method_configs)
            active.cluster_scale = cluster_scale

        wrapper = wrapper_methods.TargetOnlyWrapper(method_configs)
        if method_configs.use_knn:
            knn = method.SKLKNNRegression(deepcopy(method_configs))
            knn.cv_params['n_neighbors'] = np.asarray([1])
            wrapper.base_learner = knn
        else:
            nw = method.NadarayaWatsonMethod(deepcopy(method_configs))
            wrapper.base_learner = nw

        active.base_learner = wrapper
        active.base_learner.quiet = False
        self.learner = active

class MethodConfigs(bc.MethodConfigs):
    def __init__(self, pc):
        super(MethodConfigs, self).__init__()
        self.copy_fields(pc,pc_fields_to_copy)

class BatchConfigs(bc.BatchConfigs):
    def __init__(self, pc):
        super(BatchConfigs, self).__init__()
        from experiment.experiment_manager import MethodExperimentManager
        self.method_experiment_manager_class = MethodExperimentManager
        self.config_list = list()
        for d in data_sets_for_exps:
            pc2 = ProjectConfigs(d)

            all_active_methods = [
                ACTIVE_RANDOM,
                ACTIVE_CLUSTER,
                ACTIVE_CLUSTER_PURITY
            ]
            if not run_batch_methods:
                all_active_methods = [
                    active_method
                ]
            for i in all_active_methods:
                p = deepcopy(pc2)
                p.active_method = deepcopy(i)
                m = MainConfigs(p)
                if d in large_data_sets:
                    m.learner.max_items_for_instance_selection = max_target_items_for_large_data_sets
                self.config_list.append(m)


            # Oracle experiment
            p = deepcopy(pc2)
            p.active_method = ACTIVE_CLUSTER_PURITY
            m = MainConfigs(p)
            m.learner.use_oracle_target = True
            if d in large_data_sets:
                m.learner.max_items_for_instance_selection = max_target_items_for_large_data_sets
            self.config_list.append(m)
            '''
            # Spectral clustering without f(x) (i.e. baseline)
            p = deepcopy(pc2)
            p.active_method = ACTIVE_CLUSTER_PURITY
            p.use_greedy_instance_selection = False
            p.no_f_x = True
            p.fixed_sigma_x = True
            m = MainConfigs(p)
            if d in large_data_sets:
                m.learner.max_items_for_instance_selection = max_target_items_for_large_data_sets
            self.config_list.append(m)

            # Spectral clustering, no f(x), no kernel
            p = deepcopy(pc2)
            p.active_method = ACTIVE_CLUSTER_PURITY
            p.use_greedy_instance_selection = False
            p.no_f_x = True
            p.fixed_sigma_x = True
            p.no_spectral_kernel = True
            m = MainConfigs(p)
            if d in large_data_sets:
                m.learner.max_items_for_instance_selection = max_target_items_for_large_data_sets
            self.config_list.append(m)
            '''

class VisualizationConfigs(bc.VisualizationConfigs):
    PLOT_PAIRWISE = 1
    PLOT_BOUND = 2
    PLOT_NEIGHBOR = 3
    PLOT_SIMILAR = 4
    def __init__(self, data_set=None, **kwargs):
        super(VisualizationConfigs, self).__init__(data_set, **kwargs)
        if getattr(self, 'plot_type', None) is None:
            self.plot_type = VisualizationConfigs.PLOT_PAIRWISE

        self.max_rows = max_rows
        pc = ProjectConfigs(data_set)
        self.copy_fields(pc,pc_fields_to_copy)
        self.use_tight_layout = False
        self.figsize = (10,8.9)
        self.borders = (.05,.95,.95,.05)
        self.data_set_to_use = pc.data_set
        self.title = bc.data_name_dict.get(self.data_set_to_use, 'Unknown Data Set')
        self.show_legend_on_all = show_legend_on_all
        self.x_axis_string = 'Number of labeled instances'

        self.files = OrderedDict()
        #self.files['ActiveRandom+TargetOnlyWrapper+NW.pkl'] = 'Random NW'
        #self.files['ActiveCluster+TargetOnlyWrapper+NW.pkl'] = 'Cluster NW'
        #self.files['ActiveClusterPurity+TargetOnlyWrapper+NW.pkl'] = 'Cluster Purity NW'
        #self.files['ActiveClusterPurity-targetVar_items=%d_scale=10+TargetOnlyWrapper+NW.pkl' % active_items_per_iteration] = 'Cluster Purity NW: Target Variance'
        #self.files['ActiveClusterPurity-targetVar+TargetOnlyWrapper+NW.pkl'] = 'Cluster Purity NW: Target Variance'
        #self.files['ActiveClusterPurity-targetVar-density+TargetOnlyWrapper+NW.pkl'] = 'Cluster Purity NW: Density, Target Variance'

        '''
        self.files['ActiveClusterPurity-instanceSel_items=%d_iters=%d+TargetOnlyWrapper+NW.pkl' % (active_items_per_iteration, active_iterations)] = 'Supervised Cluster NW'
        self.files['ActiveRandom_items=%d_iters=%d+TargetOnlyWrapper+NW.pkl' % (
        active_items_per_iteration, active_iterations)] = 'Random NW'
        self.files['ActiveCluster_items=%d_iters=%d_scale=10+TargetOnlyWrapper+NW.pkl' % (
        active_items_per_iteration, active_iterations)] = 'Cluster NW'
        '''

        self.files['ActiveRandom_n=%d_items=%d_iters=2+TargetOnlyWrapper+SKL-KNNReg.pkl'] = 'Random'
        self.files['ActiveCluster_n=%d_items=%d_iters=2_scale=1+TargetOnlyWrapper+SKL-KNNReg.pkl'] = 'Cluster'
        #self.files['ActiveClusterPurity-instanceSel_n=%d_items=%d_iters=2_noY_fixedSigX+TargetOnlyWrapper+NW.pkl'] = 'Spectral Cluster'
        #self.files['ActiveClusterPurity-instanceSel_n=%d_items=%d_iters=2_targetSubsample=300_noY_fixedSigX+TargetOnlyWrapper+NW.pkl'] = 'Spectral Cluster, 300 subsample'
        '''
        self.files[
            'ActiveClusterPurity-instanceSel_n=%d_items=%d_iters=2_noY_noSpectralX+TargetOnlyWrapper+NW.pkl'] = 'Spectral Cluster'
        self.files[
            'ActiveClusterPurity-instanceSel_n=%d_items=%d_iters=2_targetSubsample=300_noY_noSpectralX+TargetOnlyWrapper+NW.pkl'] = 'Spectral Cluster, 300 subsample'
        '''
        self.files['ActiveClusterPurity-greedyInstanceSel-l1Loss_n=%d_items=%d_iters=2+TargetOnlyWrapper+SKL-KNNReg.pkl'] = 'Our Method'
        self.files['ActiveClusterPurity-greedyInstanceSel-l1Loss_n=%d_items=%d_iters=2_oracleTargetY+TargetOnlyWrapper+SKL-KNNReg.pkl'] = 'Our Method, Oracle Target Y'
        self.files['ActiveClusterPurity-greedyInstanceSel-l1Loss_n=%d_items=%d_iters=2_targetSubsample=300+TargetOnlyWrapper+SKL-KNNReg.pkl'] = 'Our Method, 300 subsample'
        self.files['ActiveClusterPurity-greedyInstanceSel-l1Loss_n=%d_items=%d_iters=2_oracleTargetY_targetSubsample=300+TargetOnlyWrapper+SKL-KNNReg.pkl'] = 'Our Method, Oracle Target Y, 300 subsample'

        '''
        self.files['ActiveClusterPurity-greedyInstanceSel_n=%d_items=%d_iters=2+TargetOnlyWrapper+NW.pkl'] = 'Our Method, Greedy'
        self.files[
            'ActiveClusterPurity-greedyInstanceSel_n=%d_items=%d_iters=2_oracleTargetY+TargetOnlyWrapper+NW.pkl'] = 'Our Method, Greedy, Oracle Target Y'
        self.files[
            'ActiveClusterPurity-greedyInstanceSel_n=%d_items=%d_iters=2_targetSubsample=300+TargetOnlyWrapper+NW.pkl'] = 'Our Method, Greedy, 300 subsample'
        self.files[
            'ActiveClusterPurity-greedyInstanceSel_n=%d_items=%d_iters=2_oracleTargetY_targetSubsample=300+TargetOnlyWrapper+NW.pkl'] = 'Our Method, Greedy, Oracle Target Y, 300 subsample'
        '''
        new_files = OrderedDict()
        for key, value in self.files.iteritems():
            new_files[key % (num_starting_labels, active_items_per_iteration)] = value
        self.files = new_files




        # self.files['ActiveClusterPurity-instanceSel_warmStart_n=2_items=5_iters=2+TargetOnlyWrapper+NW.pkl'] = 'Our Method, warm start'
        # self.files['ActiveClusterPurity-instanceSel_n=0_items=5_iters=2_oracleY+TargetOnlyWrapper+NW.pkl'] = 'Our Method, Oracle Y'

#viz_params = [dict()]
viz_params = [
    {'data_set': d} for d in all_data_sets
]