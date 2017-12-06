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

# Command line arguments for ProjectConfigs
arguments = None


def create_project_configs():
    return ProjectConfigs()

pc_fields_to_copy = bc.pc_fields_to_copy + [
    'include_size_in_file_name'
]
data_set_to_use = bc.DATA_SYNTHETIC_LINEAR_REGRESSION
#data_set_to_use = bc.DATA_BOSTON_HOUSING
#data_set_to_use = bc.DATA_CONCRETE
#data_set_to_use = bc.DATA_DROSOPHILIA

#data_set_to_use = bc.DATA_ADIENCE_ALIGNED_CNN_1

data_sets_for_exps = [data_set_to_use]

viz_for_paper = True

run_experiments = True
use_test_error_for_model_selection = False
use_validation = True

# Conference paper experiments
batch_pairwise = False
batch_neighbor = False
batch_similar = False
batch_bound = False
batch_ssl = False
batch_hinge_exps = False
batch_size = [50]

# Journal paper experiments
batch_relative_variance = False
batch_relative_bias = False
batch_relative_diversity = False
batch_relative_chain = False
batch_relative_honest = False
batch_relative_combine_guidance = True

PLOT_VARIANCE = 0
PLOT_BIAS = 1
PLOT_DIVERSITY = 2
PLOT_CHAIN = 3
PLOT_COMBINE_GUIDANCE = 4
journal_plot_type = PLOT_COMBINE_GUIDANCE

bias_scale = 0
bias_threshold = 0
mixed_guidance_set_size = 0
num_chain_instances = 0

include_size_in_file_name = False

small_param_range = False
tune_scale = False
ridge_on_fail = False

num_features = -1
other_method_configs = {
    'y_scale_min_max': False,
    'y_scale_standard': False,
    'scipy_opt_method': 'L-BFGS-B',
    'num_cv_splits': 10,
    'eps': 1e-10,
    'use_perfect_feature_selection': False
}

use_mixed_cv = False
use_ssl = False
use_mean = False
use_baseline = False

use_pairwise = False
num_pairwise = 50
logistic_noise = 0
#pair_bound = (.25,1)
pair_bound = ()
use_hinge = False
noise_rate = .0
use_logistic_fix = False
pairwise_use_scipy = True

use_bound = False
num_bound = 51
use_quartiles = True
bound_logistic = True

use_neighbor = False
num_neighbor = 51
use_min_pair_neighbor = False
fast_dccp = True
init_ridge = False
init_ideal = False
init_ridge_train = False
use_neighbor_logistic = False
neighbor_convex = False
neighbor_hinge = False
neighbor_exp = True

use_similar = False
num_similar = 51
use_similar_hinge = False
similar_use_scipy = True

use_aic = True
run_batch = True
if helper_functions.is_laptop():
    run_batch = True

active_iterations = 2
active_items_per_iteration = 50
run_active_experiments = False

show_legend_on_all = True

max_rows = 3

synthetic_dim = 1
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
        self.project_dir = 'active'
        self.use_pool = use_pool
        self.pool_size = pool_size
        if run_active_experiments:
            self.method_results_class = results_lib.ActiveMethodResults
        if data_set is None:
            data_set = data_set_to_use
        self.set_data_set(data_set)
        self.num_splits = 30
        if use_arguments and arguments is not None:
            apply_arguments(self)

        for key, value in other_method_configs.items():
            setattr(self, key, value)

        self.use_mixed_cv = use_mixed_cv
        self.use_ssl = use_ssl
        self.use_mean = use_mean
        self.use_baseline = use_baseline
        self.ridge_on_fail = ridge_on_fail
        self.tune_scale = tune_scale
        self.small_param_range = small_param_range

        self.use_pairwise = use_pairwise
        self.num_pairwise = num_pairwise
        self.pair_bound = pair_bound
        self.use_hinge = use_hinge
        self.noise_rate = noise_rate
        self.logistic_noise = logistic_noise
        self.pairwise_use_scipy = pairwise_use_scipy
        self.bias_scale = bias_scale
        self.bias_threshold = bias_threshold
        self.mixed_guidance_set_size = mixed_guidance_set_size
        self.num_chain_instances = num_chain_instances


        self.use_bound = use_bound
        self.num_bound = num_bound
        self.use_quartiles = use_quartiles
        self.bound_logistic = bound_logistic

        self.use_neighbor = use_neighbor
        self.num_neighbor = num_neighbor
        self.use_min_pair_neighbor = use_min_pair_neighbor
        self.fast_dccp = fast_dccp
        self.init_ridge = init_ridge
        self.init_ideal = init_ideal
        self.init_ridge_train = init_ridge_train
        self.use_neighbor_logistic = use_neighbor_logistic
        self.use_logistic_fix = use_logistic_fix
        self.neighbor_convex = neighbor_convex
        self.neighbor_hinge = neighbor_hinge
        self.neighbor_exp = neighbor_exp

        self.num_features = num_features
        if self.data_set in {bc.DATA_DROSOPHILIA}:
            self.num_features = 50

        self.use_similar = use_similar
        self.num_similar = num_similar
        self.use_similar_hinge = use_similar_hinge
        self.similar_use_scipy = similar_use_scipy

        self.use_test_error_for_model_selection = use_test_error_for_model_selection
        self.use_aic = use_aic

        self.include_size_in_file_name = include_size_in_file_name

    def set_data_set(self, data_set):
        self.data_set = data_set
        if data_set == bc.DATA_BOSTON_HOUSING:
            self.set_boston_housing()
            self.num_labels = [5, 10, 20, 40]
            if run_active_experiments:
                self.num_labels = [5]
        elif data_set == bc.DATA_SYNTHETIC_LINEAR_REGRESSION:
            self.set_synthetic_linear_reg()
            self.num_labels = [10, 20, 40]
            if run_active_experiments:
                self.num_labels = [20]
        elif data_set == bc.DATA_ADIENCE_ALIGNED_CNN_1:
            self.set_adience_aligned_cnn_1()
            #self.num_labels = [5]
            #self.num_labels = [10, 20, 40, 80]
            self.num_labels = [10, 20, 40]
            if run_active_experiments:
                self.num_labels = [20]
        elif data_set == bc.DATA_WINE_RED:
            self.set_wine_red()
            self.num_labels = [5, 10, 20, 40]
            #self.num_labels = [5]
            if run_active_experiments:
                self.num_labels = [20]
        elif data_set == bc.DATA_CONCRETE:
            self.set_concrete()
            self.num_labels = [5, 10, 20, 40]
        elif data_set == bc.DATA_DROSOPHILIA:
            self.set_drosophilia()
            self.num_labels = [10,20,40]
        '''
        if self.include_size_in_file_name:
            assert len(self.num_labels) == 1
        '''

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
        method_configs.use_validation = use_validation

        for key in other_method_configs.keys():
            setattr(method_configs, key, getattr(pc,key))

        method_configs.use_mixed_cv = pc.use_mixed_cv
        method_configs.use_baseline = pc.use_baseline
        method_configs.ridge_on_fail = pc.ridge_on_fail
        method_configs.tune_scale = pc.tune_scale
        method_configs.small_param_range = pc.small_param_range

        method_configs.use_pairwise = pc.use_pairwise
        method_configs.num_pairwise = pc.num_pairwise
        method_configs.pair_bound = pc.pair_bound
        method_configs.use_hinge = pc.use_hinge
        method_configs.noise_rate = pc.noise_rate
        method_configs.logistic_noise = pc.logistic_noise
        method_configs.pairwise_use_scipy = pc.pairwise_use_scipy
        method_configs.bias_scale = pc.bias_scale
        method_configs.bias_threshold = pc.bias_threshold
        method_configs.mixed_guidance_set_size = pc.mixed_guidance_set_size
        method_configs.num_chain_instances = pc.num_chain_instances

        method_configs.use_bound = pc.use_bound
        method_configs.num_bound = pc.num_bound
        method_configs.use_quartiles = pc.use_quartiles
        method_configs.bound_logistic = pc.bound_logistic

        method_configs.use_neighbor = pc.use_neighbor
        method_configs.num_neighbor = pc.num_neighbor
        method_configs.use_min_pair_neighbor = pc.use_min_pair_neighbor
        method_configs.fast_dccp = pc.fast_dccp
        method_configs.init_ridge = pc.init_ridge
        method_configs.init_ideal = pc.init_ideal
        method_configs.init_ridge_train = pc.init_ridge_train
        method_configs.use_neighbor_logistic = pc.use_neighbor_logistic
        method_configs.use_logistic_fix = pc.use_logistic_fix
        method_configs.neighbor_convex = pc.neighbor_convex
        method_configs.neighbor_hinge = pc.neighbor_hinge
        method_configs.neighbor_exp = pc.neighbor_exp

        method_configs.use_similar = pc.use_similar
        method_configs.num_similar = pc.num_similar
        method_configs.use_similar_hinge = pc.use_similar_hinge
        method_configs.similar_use_scipy = pc.similar_use_scipy

        method_configs.use_test_error_for_model_selection = pc.use_test_error_for_model_selection
        method_configs.use_aic = pc.use_aic
        method_configs.num_features = pc.num_features
        #active = active_methods.ActiveMethod(method_configs)
        active = active_methods.RelativeActiveMethod(method_configs)
        active.base_learner = methods.method.RelativeRegressionMethod(method_configs)
        relative_reg = methods.method.RelativeRegressionMethod(method_configs)
        ridge_reg = method.SKLRidgeRegression(method_configs)
        mean_reg = method.SKLMeanRegressor(method_configs)

        lap_ridge = semisupervised.LaplacianRidgeMethod(method_configs)
        if run_active_experiments:
            self.learner = active
        else:
            if pc.use_ssl:
                self.learner = lap_ridge
            elif pc.use_mean:
                self.learner = mean_reg
            else:
                self.learner = relative_reg

class MethodConfigs(bc.MethodConfigs):
    def __init__(self, pc):
        super(MethodConfigs, self).__init__()
        self.copy_fields(pc,pc_fields_to_copy)

class BatchConfigs(bc.BatchConfigs):
    def __init__(self, pc):
        super(BatchConfigs, self).__init__()
        from experiment.experiment_manager import MethodExperimentManager
        self.method_experiment_manager_class = MethodExperimentManager
        if not run_batch:
            self.config_list = [MainConfigs(pc)]
            return

        c = pc.copy()
        c.use_pairwise = False
        c.use_neighbor = False
        c.use_bound = False
        c.use_hinge = False
        c.use_quartile = False
        c.use_similar = False
        c.use_similar_hinge = False
        #c.use_test_error_for_model_selection = False
        self.config_list = [MainConfigs(c)]

        if batch_ssl:
            ssl_params = {
                'use_ssl': [True]
            }
            self.config_list += [MainConfigs(configs) for configs in c.generate_copies(ssl_params)]

        if batch_pairwise:
            pairwise_params = {
                'use_pairwise': [True],
                'num_pairwise': batch_size,
            }
            self.config_list += [MainConfigs(configs) for configs in c.generate_copies(pairwise_params)]
            if batch_hinge_exps:
                pairwise_hinge_params = {
                    'use_pairwise': [True],
                    'use_hinge': [True],
                    'num_pairwise': batch_size,
                }
                self.config_list += [MainConfigs(configs) for configs in c.generate_copies(pairwise_hinge_params)]

        if batch_bound:
            bound_params = {
                'use_bound': [True],
                'num_bound': batch_size,
            }
            self.config_list += [MainConfigs(configs) for configs in c.generate_copies(bound_params)]

            bound_baseline_params = {
                'use_bound': [True],
                'num_bound': batch_size,
                'use_baseline': [True],
                'bound_logistic': [False]
            }

            self.config_list += [MainConfigs(configs) for configs in c.generate_copies(bound_baseline_params)]
            '''
            if batch_hinge_exps:
                bound_hinge_params = {
                    'use_bound': [True],
                    'num_bound': batch_size,
                    'bound_logistic': [False]
                }
                self.config_list += [MainConfigs(configs) for configs in c.generate_copies(bound_hinge_params)]
            '''
        if batch_neighbor:
            neighbor_params = {
                'use_neighbor': [True],
                'num_neighbor': batch_size,
            }
            self.config_list += [MainConfigs(configs) for configs in c.generate_copies(neighbor_params)]
            '''
            if batch_hinge_exps:
                neighbor_hinge_params = {
                    'use_neighbor': [True],
                    'neighbor_hinge': [True],
                    'num_neighbor': batch_size,
                }
                self.config_list += [MainConfigs(configs) for configs in c.generate_copies(neighbor_hinge_params)]
            '''
        if batch_similar:
            similar_params = {
                'use_similar': [True],
                'num_similar': batch_size,
            }
            self.config_list += [MainConfigs(configs) for configs in c.generate_copies(similar_params)]
            '''
            if batch_hinge_exps:
                similar_hinge_params = {
                    'use_similar': [True],
                    'use_similar_hinge': [True],
                    'num_similar': batch_size,
                }
                self.config_list += [MainConfigs(configs) for configs in c.generate_copies(similar_hinge_params)]
            '''
        if batch_relative_variance:
            pairwise_params = {
                'use_pairwise': [True],
                'num_pairwise': batch_size,
            }
            self.config_list += [MainConfigs(configs) for configs in c.generate_copies(pairwise_params)]
            pairwise_params['logistic_noise'] = [.1, .2]
            self.config_list += [MainConfigs(configs) for configs in c.generate_copies(pairwise_params)]
        if batch_relative_bias:
            bias_params = {
                'bias_scale': [.05, .1, .2],
                'bias_threshold': [10],
            }
            self.config_list += [MainConfigs(configs) for configs in c.generate_copies(bias_params)]
        if batch_relative_diversity:
            pairwise_params = {
                'use_pairwise': [True],
                'num_pairwise': batch_size,
                'mixed_guidance_set_size': [10, 20, 40, 80, 160]
            }
            self.config_list += [MainConfigs(configs) for configs in c.generate_copies(pairwise_params)]
        if batch_relative_chain:
            pairwise_params = {
                'use_pairwise': [True],
                'num_chain_instances': [1, 5, 10],
            }
            self.config_list += [MainConfigs(configs) for configs in c.generate_copies(pairwise_params)]
        if batch_relative_honest:
            pass
        if batch_relative_combine_guidance:
            combined_params = {
                'use_pairwise': [True],
                'num_pairwise': [25],
                'use_similar': [True],
                'num_similar': [25],
            }
            self.config_list += [MainConfigs(configs) for configs in c.generate_copies(combined_params)]
            pairwise_params = {
                'use_pairwise': [True],
                'num_pairwise': [25],
            }
            self.config_list += [MainConfigs(configs) for configs in c.generate_copies(pairwise_params)]
            similar_params = {
                'use_similar': [True],
                'num_similar': [25],
            }
            self.config_list += [MainConfigs(configs) for configs in c.generate_copies(similar_params)]

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
            self.ylims = [0,1]
        elif pc.data_set == bc.DATA_ADIENCE_ALIGNED_CNN_1:
            self.ylims = [0,1]
        elif pc.data_set == bc.DATA_BOSTON_HOUSING:
            self.ylims = [0,1]
        elif pc.data_set == bc.DATA_CONCRETE:
            self.ylims = [0,1]
        elif pc.data_set == bc.DATA_DROSOPHILIA:
            self.ylims = [0,1]
        self.ylims = None
        self.files = OrderedDict()
        if run_active_experiments:
            self.files['RelActiveRandom+SKL-RidgeReg.pkl'] = 'Random Pairwise, SKLRidge'
            self.files['ActiveRandom+SKL-RidgeReg.pkl'] = 'Random, SKLRidge'
            self.files['RelActiveRandom+RelReg-cvx-log-with-log-noLinear-TEST.pkl'] = 'TEST: RandomPairwise, RelReg'
        else:
            self.generate_file_names(pc)




    def generate_file_names(self, pc):
        self.files = OrderedDict()
        base_file_name = 'RelReg-cvx-constraints-%s=%s'
        use_test = use_test_error_for_model_selection
        if pc.num_features > 0:
            if use_test:
                self.files['RelReg-cvx-constraints-noPairwiseReg-numFeats=' + str(pc.num_features) + '-TEST.pkl'] = 'TEST: Ridge Regression'
                self.files['RelReg-cvx-constraints-noPairwiseReg-numFeats=' + str(pc.num_features) + '-nCV=10.pkl'] = 'Ridge Regression'
            else:
                self.files['RelReg-cvx-constraints-noPairwiseReg-numFeats=' + str(
                    pc.num_features) + '-nCV=10-VAL.pkl'] = 'Ridge Regression'
                if journal_plot_type == PLOT_VARIANCE:
                    self.files[
                        'RelReg-cvx-constraints-numRandPairs=50-scipy-logNoise=0.1-noRidgeOnFail-solver=SCS-numFeats=50-L-BFGS-B-nCV=10-VAL.pkl'] = '50 pairs, .1 noise'
                    self.files[
                        'RelReg-cvx-constraints-numRandPairs=50-scipy-logNoise=0.2-noRidgeOnFail-solver=SCS-numFeats=50-L-BFGS-B-nCV=10-VAL.pkl'] = '50 pairs, .2 noise'
                if journal_plot_type == PLOT_BIAS:
                    self.files[
                        'RelReg-cvx-constraints-noPairwiseReg-numFeats=50-nCV=10-biasThresh=10-biasScale=0.2-VAL.pkl'] = 'Ridge, biasScale=.2'
                    self.files[
                        'RelReg-cvx-constraints-noPairwiseReg-numFeats=50-nCV=10-biasThresh=10-biasScale=0.1-VAL.pkl'] = 'Ridge, biasScale=.1'
                    self.files[
                        'RelReg-cvx-constraints-noPairwiseReg-numFeats=50-nCV=10-biasThresh=10-biasScale=0.05-VAL.pkl'] = 'Ridge, biasScale=.05'
                if journal_plot_type == PLOT_DIVERSITY:
                    self.files['RelReg-cvx-constraints-numRandPairs=50-scipy-noRidgeOnFail-solver=SCS-numFeats=50-L-BFGS-B-nCV=10-setSize=10-VAL.pkl'] = '50 pairs, set size 10'
                    self.files[
                        'RelReg-cvx-constraints-numRandPairs=50-scipy-noRidgeOnFail-solver=SCS-numFeats=50-L-BFGS-B-nCV=10-setSize=20-VAL.pkl'] = '50 pairs, set size 20'
                    self.files[
                        'RelReg-cvx-constraints-numRandPairs=50-scipy-noRidgeOnFail-solver=SCS-numFeats=50-L-BFGS-B-nCV=10-setSize=40-VAL.pkl'] = '50 pairs, set size 40'
                    self.files[
                        'RelReg-cvx-constraints-numRandPairs=50-scipy-noRidgeOnFail-solver=SCS-numFeats=50-L-BFGS-B-nCV=10-setSize=80-VAL.pkl'] = '50 pairs, set size 80'
                if journal_plot_type == PLOT_CHAIN:
                    self.files['/RelReg-cvx-constraints-numRandPairs=50-scipy-noRidgeOnFail-solver=SCS-numFeats=50-L-BFGS-B-nCV=10-numChains=1-VAL.pkl'] = '50 pairs, 1 chain'
                    self.files[
                        '/RelReg-cvx-constraints-numRandPairs=50-scipy-noRidgeOnFail-solver=SCS-numFeats=50-L-BFGS-B-nCV=10-numChains=5-VAL.pkl'] = '50 pairs, 5 chains'
                    self.files[
                        '/RelReg-cvx-constraints-numRandPairs=50-scipy-noRidgeOnFail-solver=SCS-numFeats=50-L-BFGS-B-nCV=10-numChains=10-VAL.pkl'] = '50 pairs, 10 chains'
                if journal_plot_type == PLOT_COMBINE_GUIDANCE:
                    self.files['RelReg-cvx-constraints-noPairwiseReg-nCV=10-VAL.pkl'] = 'Ridge'
                    self.files[
                        'RelReg-cvx-constraints-numRandPairs=25-scipy-numSimilar=25-scipy-noRidgeOnFail-eps=1e-10-solver=SCS-numFeats=50-L-BFGS-B-nCV=10-VAL.pkl'] = '25 similar, 25 pairs'
                    self.files[
                        'RelReg-cvx-constraints-numRandPairs=25-scipy-noRidgeOnFail-solver=SCS-numFeats=50-L-BFGS-B-nCV=10-VAL.pkl'] = '25 pairs'
                    self.files[
                        'RelReg-cvx-constraints-numSimilar=25-scipy-noRidgeOnFail-eps=1e-10-solver=SCS-numFeats=50-L-BFGS-B-nCV=10-VAL.pkl'] = '25 similar'

        else:
            if use_test:
                self.files['RelReg-cvx-constraints-noPairwiseReg-TEST.pkl'] = 'TEST: Ridge Regression'
                self.files['RelReg-cvx-constraints-noPairwiseReg.pkl'] = 'Ridge Regression'
            else:
                self.files['RelReg-cvx-constraints-noPairwiseReg-nCV=10-VAL.pkl'] = 'Ridge Regression'
                if journal_plot_type == PLOT_VARIANCE:
                    self.files[
                        'RelReg-cvx-constraints-numRandPairs=50-scipy-logNoise=0.1-noRidgeOnFail-solver=SCS-L-BFGS-B-nCV=10-VAL.pkl'] = '50 pairs, .1 noise'
                    self.files[
                        'RelReg-cvx-constraints-numRandPairs=50-scipy-logNoise=0.2-noRidgeOnFail-solver=SCS-L-BFGS-B-nCV=10-VAL.pkl'] = '50 pairs, .2 noise'
                if journal_plot_type == PLOT_BIAS:
                    self.files[
                        'RelReg-cvx-constraints-noPairwiseReg-nCV=10-biasThresh=10-biasScale=0.2-VAL.pkl'] = 'Ridge, biasScale=.2'
                    self.files['RelReg-cvx-constraints-noPairwiseReg-nCV=10-biasThresh=10-biasScale=0.1-VAL.pkl'] = 'Ridge, biasScale=.1'
                    self.files[
                        'RelReg-cvx-constraints-noPairwiseReg-nCV=10-biasThresh=10-biasScale=0.05-VAL.pkl'] = 'Ridge, biasScale=.05'
                if journal_plot_type == PLOT_DIVERSITY:
                    self.files['RelReg-cvx-constraints-numRandPairs=50-scipy-noRidgeOnFail-solver=SCS-L-BFGS-B-nCV=10-setSize=10-VAL.pkl'] = '50 pairs, set size 10'
                    self.files[
                        'RelReg-cvx-constraints-numRandPairs=50-scipy-noRidgeOnFail-solver=SCS-L-BFGS-B-nCV=10-setSize=20-VAL.pkl'] = '50 pairs, set size 20'
                    self.files[
                        'RelReg-cvx-constraints-numRandPairs=50-scipy-noRidgeOnFail-solver=SCS-L-BFGS-B-nCV=10-setSize=40-VAL.pkl'] = '50 pairs, set size 40'
                    self.files[
                        'RelReg-cvx-constraints-numRandPairs=50-scipy-noRidgeOnFail-solver=SCS-L-BFGS-B-nCV=10-setSize=80-VAL.pkl'] = '50 pairs, set size 80'
                if journal_plot_type == PLOT_CHAIN:
                    self.files[
                        'RelReg-cvx-constraints-numRandPairs=50-scipy-noRidgeOnFail-solver=SCS-L-BFGS-B-nCV=10-numChains=1-VAL.pkl'] = '50 pairs, 1 chain'
                    self.files[
                        'RelReg-cvx-constraints-numRandPairs=50-scipy-noRidgeOnFail-solver=SCS-L-BFGS-B-nCV=10-numChains=5-VAL.pkl'] = '50 pairs, 5 chain'
                    self.files[
                        'RelReg-cvx-constraints-numRandPairs=50-scipy-noRidgeOnFail-solver=SCS-L-BFGS-B-nCV=10-numChains=10-VAL.pkl'] = '50 pairs, 10 chain'
                if journal_plot_type == PLOT_COMBINE_GUIDANCE:
                    self.files['RelReg-cvx-constraints-noPairwiseReg-nCV=10-VAL.pkl'] = 'Ridge'
                    self.files[
                        'RelReg-cvx-constraints-numRandPairs=25-scipy-numSimilar=25-scipy-noRidgeOnFail-eps=1e-10-solver=SCS-L-BFGS-B-nCV=10-VAL.pkl'] = '25 similar, 25 pairs'
                    self.files[
                        'RelReg-cvx-constraints-numRandPairs=25-scipy-noRidgeOnFail-solver=SCS-L-BFGS-B-nCV=10-VAL.pkl'] = '25 pairs'
                    self.files[
                        'RelReg-cvx-constraints-numSimilar=25-scipy-noRidgeOnFail-eps=1e-10-solver=SCS-L-BFGS-B-nCV=10-VAL.pkl'] = '25 similar'
        self.files['LapRidge-VAL.pkl'] = 'Laplacian Ridge Regression'
        #self.files['SKL-DumReg.pkl'] = 'Predict Mean'
        sizes = []
        #sizes.append(10)
        #sizes.append(20)
        sizes.append(50)
        #sizes.append(100)
        #sizes.append(150)
        #sizes.append(250)
        suffixes = OrderedDict()
        #suffixes['pairBound'] = [(0,.1),(0,.25),(0,.5),(0,.75),None]
        #suffixes['pairBound'] = [(.5,1), (.25,1), None]
        #suffixes['mixedCV'] = [None,'']
        #suffixes['logNoise'] = [None, .1, .5, 1, 2]
        #suffixes['logNoise'] = [.5]
        #suffixes['logNoise'] = [None,25,50,100]
        suffixes['baseline'] = [None,'']
        #suffixes['noGrad'] = ['']
        if pc.num_features > 0:
            if other_method_configs['use_perfect_feature_selection']:
                suffixes['numFeatsPerfect'] = [str(pc.num_features)]
            else:
                suffixes['numFeats'] = [str(pc.num_features)]
        suffixes['scipy'] = [None, '']
        suffixes['noRidgeOnFail'] = [None, '']
        suffixes['tuneScale'] = [None, '']
        #suffixes['smallScale'] = [None, '']
        #suffixes['minMax'] = [None, '']
        #suffixes['zScore'] = [None, '']
        suffixes['solver'] = ['SCS']
        suffixes['L-BFGS-B'] = [None, '']
        #suffixes['logNoise'] = [None, '0.01']
        if not use_test:
            suffixes['nCV'] = ['10']
        suffixes['VAL'] = ['']

        #suffixes['numFeats'] = [str(num_feat)]

        ordered_keys = [
            'fastDCCP', 'initRidge', 'init_ideal', 'initRidgeTrain','logistic',
            'pairBound', 'mixedCV', 'logNoise', 'scipy', 'logNoise', 'noGrad',
            'baseline', 'logFix', 'noRidgeOnFail', 'tuneScale',
            'smallScale', 'eps',
            'solver', 'minMax', 'zScore', 'numFeats', 'numFeatsPerfect', 'L-BFGS-B', 'nCV', 'VAL'
        ]

        methods = []
        if self.plot_type == VisualizationConfigs.PLOT_PAIRWISE:
            methods.append(('numRandPairs','RelReg, %s pairs', 'Our Method: %s relative'))
            methods.append(('numRandPairsHinge','RelReg, %s pairs hinge', 'Zhu 2007: %s relative'))
            self.title = 'Relative'
            #if pc.data_set == bc.DATA_SYNTHETIC_LINEAR_REGRESSION:
            #    self.ylims = [0,12]
        elif self.plot_type == VisualizationConfigs.PLOT_BOUND:
            methods.append(('numRandLogBounds', '%s log bounds', 'Our Method: %s bound'))
            methods.append(('numRandQuartiles', 'RelReg, %s quartiles', 'Baseline: %s'))
            suffixes['eps'] = [None, '1e-08', '1e-10', '1e-16']
            self.title = 'Bound'
        elif self.plot_type == VisualizationConfigs.PLOT_NEIGHBOR:
            #methods.append(('numRandNeighborConvex', 'RelReg, %s rand neighbors convex', 'Our Method: %s neighbors'))
            methods.append(('numRandPairs','RelReg, %s pairs', 'Our Method: %s relative'))
            #methods.append(('numRandNeighborConvexHinge', 'RelReg, %s rand neighbors convex hinge', 'Our Method: %s neighbor, hinge'))
            methods.append(('numRandNeighborExp', 'RelReg, %s rand neighbors convex exp', 'Our Method: %s neighbor'))
            sizes = [20, 50]
            self.title = 'Neighbor'
        elif self.plot_type == VisualizationConfigs.PLOT_SIMILAR:
            suffixes['eps'] = [None, '1e-08', '1e-10', '1e-16']
            methods.append(('numSimilar','RelReg, %s similar', 'Our Method: %s similar'))
            #methods.append(('numSimilarHinge','RelReg, %s pairs hinge', 'Our Method: %s similar, hinge'))
            self.title = 'Similar'

        all_params = list(grid_search.ParameterGrid(suffixes))
        for file_suffix, legend_name, legend_name_paper in methods:
            for size in sizes:
                for params in all_params:
                    file_name = base_file_name
                    file_name = file_name % (file_suffix, str(size))
                    legend = legend_name
                    if viz_for_paper:
                        legend = legend_name_paper
                    legend = legend % str(size)
                    for key in ordered_keys:
                        if not params.has_key(key):
                            continue
                        value = params[key]
                        if value is None:
                            continue
                        if value == '':
                            file_name += '-' + key
                            if not viz_for_paper:
                                legend += ', ' + key
                        else:
                            file_name += '-' + key + '=' + str(value)
                            if not viz_for_paper:
                                legend += ', ' + str(value) + ' ' + key
                    if use_test:
                        file_name += '-TEST'
                        legend = 'TEST: ' + legend
                    file_name += '.pkl'
                    self.files[file_name] = legend


viz_params = [
    {'data_set': bc.DATA_SYNTHETIC_LINEAR_REGRESSION},
    {'data_set': bc.DATA_BOSTON_HOUSING},
    {'data_set': bc.DATA_CONCRETE},
    {'data_set': bc.DATA_DROSOPHILIA},
    {'data_set': bc.DATA_ADIENCE_ALIGNED_CNN_1},
]

#For plotting all four types of guidance for a single data set
'''
viz_params = [
    {'plot_type': VisualizationConfigs.PLOT_PAIRWISE},
    {'plot_type': VisualizationConfigs.PLOT_BOUND},
    {'plot_type': VisualizationConfigs.PLOT_NEIGHBOR},
    {'plot_type': VisualizationConfigs.PLOT_SIMILAR},
]
'''