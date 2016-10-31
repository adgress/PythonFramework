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
# Command line arguments for ProjectConfigs
arguments = None


def create_project_configs():
    return ProjectConfigs()

pc_fields_to_copy = bc.pc_fields_to_copy + [
    'include_size_in_file_name'
]
data_set_to_use = bc.DATA_SYNTHETIC_LINEAR_REGRESSION
#data_set_to_use = bc.DATA_SYNTHETIC_LINEAR_REGRESSION_10
#data_set_to_use = bc.DATA_BOSTON_HOUSING
#data_set_to_use = bc.DATA_CONCRETE
#data_set_to_use = bc.DATA_DROSOPHILIA

#data_set_to_use = bc.DATA_ADIENCE_ALIGNED_CNN_1

data_sets_for_exps = [data_set_to_use]

viz_for_paper = True

run_experiments = True
use_test_error_for_model_selection = False

use_relative = True
use_pairwise_active = True

include_size_in_file_name = False

other_method_configs = {
    'y_scale_min_max': False,
    'y_scale_standard': False,
    'scipy_opt_method': 'L-BFGS-B',
    'num_cv_splits': 10,
    'eps': 1e-10,
    'use_perfect_feature_selection': True,
    'use_test_error_for_model_selection': False,
    'use_validation': True,
    'use_uncertainty': False,
    'use_oed': True,
    'num_features': None,
    'num_pairwise': 0,
    'use_true_y': True
}
if data_set_to_use in {bc.DATA_DROSOPHILIA, bc.DATA_ADIENCE_ALIGNED_CNN_1}:
    other_method_configs['num_features'] = 50

run_batch = False
if helper_functions.is_laptop():
    run_batch = False

active_iterations = 10
active_items_per_iteration = 5

show_legend_on_all = True

max_rows = 3

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
        self.project_dir = 'active_base'
        self.use_pool = use_pool
        self.pool_size = pool_size
        self.method_results_class = results_lib.ActiveMethodResults
        if data_set is None:
            data_set = data_set_to_use
        self.set_data_set(data_set)
        self.num_splits = 30
        if use_arguments and arguments is not None:
            apply_arguments(self)

        for key, value in other_method_configs.items():
            setattr(self, key, value)

        self.use_test_error_for_model_selection = use_test_error_for_model_selection

        self.include_size_in_file_name = include_size_in_file_name

    def set_data_set(self, data_set):
        self.data_set = data_set
        if data_set == bc.DATA_BOSTON_HOUSING:
            self.set_boston_housing()
            self.num_labels = [5]
        elif data_set == bc.DATA_SYNTHETIC_LINEAR_REGRESSION_10:
            self.set_synthetic_linear_reg_10()
            self.num_labels = [10]
        elif data_set == bc.DATA_SYNTHETIC_LINEAR_REGRESSION:
            self.set_synthetic_linear_reg()
            self.num_labels = [10]
        elif data_set == bc.DATA_ADIENCE_ALIGNED_CNN_1:
            self.set_adience_aligned_cnn_1()
            self.num_labels = [10]
        elif data_set == bc.DATA_WINE_RED:
            self.set_wine_red()
            self.num_labels = [20]
        elif data_set == bc.DATA_CONCRETE:
            self.set_concrete()
            self.num_labels = [10]
        elif data_set == bc.DATA_DROSOPHILIA:
            self.set_drosophilia()
            self.num_labels = [10]

    def set_drosophilia(self):
        self.loss_function = loss_function.MeanSquaredError()
        self.cv_loss_function = loss_function.MeanSquaredError()
        self.data_dir = 'data_sets/drosophilia'
        self.data_name = 'drosophilia'
        self.results_dir = 'drosophilia'
        self.data_set_file_name = 'split_data.pkl'

    def set_concrete(self):
        self.loss_function = loss_function.MeanSquaredError()
        self.cv_loss_function = loss_function.MeanSquaredError()
        self.data_dir = 'data_sets/concrete'
        self.data_name = 'concrete'
        self.results_dir = 'concrete'
        self.data_set_file_name = 'split_data.pkl'

    def set_boston_housing(self):
        self.loss_function = loss_function.MeanSquaredError()
        self.cv_loss_function = loss_function.MeanSquaredError()
        self.data_dir = 'data_sets/boston_housing'
        self.data_name = 'boston_housing'
        self.results_dir = 'boston_housing'
        self.data_set_file_name = 'split_data.pkl'

    def set_synthetic_linear_reg_10(self):
        self.loss_function = loss_function.MeanSquaredError()
        self.cv_loss_function = loss_function.MeanSquaredError()
        self.data_dir = 'data_sets/synthetic_linear_reg500-10-1.01'
        self.data_name = 'synthetic_linear_reg500-10-1.01'
        self.results_dir = 'synthetic_linear_reg500-10-1.01'
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

    def set_wine_red(self):
        self.loss_function = loss_function.MeanSquaredError()
        self.cv_loss_function = loss_function.MeanSquaredError()
        s = 'wine-red'
        self.data_dir = 'data_sets/' + s
        self.data_name = s
        self.results_dir = s
        self.data_set_file_name = 'split_data.pkl'


class MainConfigs(bc.MainConfigs):
    def __init__(self, pc):
        super(MainConfigs, self).__init__()
        #pc = create_project_configs()
        self.copy_fields(pc,pc_fields_to_copy)
        from methods import method
        from methods import active_methods
        from methods import semisupervised
        method_configs = MethodConfigs(pc)
        method_configs.active_iterations = active_iterations
        method_configs.active_items_per_iteration = active_items_per_iteration
        method_configs.metric = 'euclidean'
        method_configs.small_param_range = False
        method_configs.num_features = -1
        method_configs.use_perfect_feature_selection = False
        method_configs.use_mixed_cv = False
        method_configs.use_baseline = False

        for key in other_method_configs.keys():
            setattr(method_configs, key, getattr(pc,key))

        use_oed = method_configs.use_oed
        use_uncertainty = method_configs.use_uncertainty
        method_configs.use_test_error_for_model_selection = pc.use_test_error_for_model_selection

        if use_pairwise_active:
            if use_oed:
                active = active_methods.RelativeActiveOEDMethod(method_configs)
            elif use_uncertainty:
                active = active_methods.RelativeActiveUncertaintyMethod(method_configs)
            else:
                active = active_methods.RelativeActiveMethod(method_configs)

        else:
            if use_oed:
                active = active_methods.OEDLinearActiveMethod(method_configs)
            else:
                active = active_methods.ActiveMethod(method_configs)

        relative_reg = methods.method.RelativeRegressionMethod(method_configs)
        ridge_reg = method.SKLRidgeRegression(method_configs)
        mean_reg = method.SKLMeanRegressor(method_configs)
        if use_relative:
            active.base_learner = relative_reg
        else:
            active.base_learner = ridge_reg


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
        if run_batch:
            new_params = [
                {'use_oed': False},
            ]
            if use_relative:
                new_params = [
                    {'use_oed': False, 'use_uncertainty': False},
                    {'use_oed': True, 'use_uncertainty': False},
                    {'use_oed': False, 'use_uncertainty': True},
                ]
        else:
            new_params = [{'unused': None}]
        self.config_list = list()
        for params in new_params:
            p = deepcopy(pc)
            p.set(**params)
            self.config_list.append(MainConfigs(p))


class VisualizationConfigs(bc.VisualizationConfigs):
    PLOT_PAIRWISE = 1
    PLOT_BOUND = 2
    PLOT_NEIGHBOR = 3
    PLOT_SIMILAR = 4
    def __init__(self, data_set=None, **kwargs):
        super(VisualizationConfigs, self).__init__(data_set, **kwargs)
        if getattr(self, 'plot_type', None) is None:
            self.plot_type = VisualizationConfigs.PLOT_PAIRWISE
        self.max_rows = 2
        pc = ProjectConfigs(data_set)
        self.copy_fields(pc,pc_fields_to_copy)

        self.figsize = (10,8.9)
        self.borders = (.05,.95,.95,.05)
        self.data_set_to_use = pc.data_set
        self.title = bc.data_name_dict.get(self.data_set_to_use, 'Unknown Data Set')
        self.show_legend_on_all = show_legend_on_all
        self.x_axis_string = 'Number of labeled instances'
        if pc.data_set == bc.DATA_SYNTHETIC_LINEAR_REGRESSION:
            self.ylims = [0,12]
        elif pc.data_set == bc.DATA_ADIENCE_ALIGNED_CNN_1:
            self.ylims = [0,1000]
        elif pc.data_set == bc.DATA_BOSTON_HOUSING:
            self.ylims = [0,200]
        elif pc.data_set == bc.DATA_CONCRETE:
            self.ylims = [0,1000]
        elif pc.data_set == bc.DATA_DROSOPHILIA:
            self.ylims = [0,3]

        self.files = OrderedDict()
        self.files['ActiveRandom+SKL-RidgeReg.pkl'] = 'Random, Ridge'
        #self.files['OED+SKL-RidgeReg.pkl'] = 'OED, Ridge'
        #self.files['OED+SKL-RidgeReg_use-labeled.pkl'] = 'OED, Ridge, use_labeled'
        if other_method_configs['num_features'] is None:
            num_feats = ''
        else:
            num_feats = '-numFeatsPerfect=' + str(other_method_configs['num_features'])
        active_opts_stf = '-10-10-5'
        rand_pairs_str = '-numRandPairs=1'
        rand_pairs_str = '-numRandPairs=0'
        files = [
            ('RelActiveRandom%s+RelReg-cvx-constraints' + rand_pairs_str + '-scipy-logFix-solver=SCS%s-L-BFGS-B-nCV=10.pkl', 'Random, pairwise, Relative=0'),
            ('RelActiveUncer%s+RelReg-cvx-constraints' + rand_pairs_str + '-scipy-logFix-solver=SCS%s-L-BFGS-B-nCV=10.pkl', 'Uncertainty, pairwise, Relative=0'),
            ('RelActiveOED%s+RelReg-cvx-constraints' + rand_pairs_str + '-scipy-logFix-solver=SCS%s-L-BFGS-B-nCV=10.pkl', 'OED, pairwise, Relative=0'),
            ('RelActiveOED-grad%s+RelReg-cvx-constraints' + rand_pairs_str + '-scipy-logFix-solver=SCS%s-L-BFGS-B-nCV=10.pkl','OED-grad, pairwise, Relative=0'),
            ('RelActiveOED-grad-labeled-%s+RelReg-cvx-constraints' + rand_pairs_str + '-scipy-logFix-solver=SCS%s-L-BFGS-B-nCV=10.pkl','OED-grad-labeled, pairwise, Relative=0'),
            ('RelActiveOED-labeled%s+RelReg-cvx-constraints' + rand_pairs_str + '-scipy-logFix-solver=SCS%s-L-BFGS-B-nCV=10.pkl','OED-labeled, pairwise, Relative=0'),
            ('RelActiveOED-E%s+RelReg-cvx-constraints' + rand_pairs_str + '-scipy-logFix-solver=SCS%s-L-BFGS-B-nCV=10.pkl','OED-E-grad, pairwise, Relative=0'),
        ]
        for file, legend in files:
            file = file % (active_opts_stf, num_feats)
            self.files[file] = legend

viz_params = [dict()]