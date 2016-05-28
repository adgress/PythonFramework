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

#Command line arguments for ProjectConfigs
arguments = None
def create_project_configs():
    return ProjectConfigs()

pc_fields_to_copy = bc.pc_fields_to_copy + [
]
#data_set_to_use = bc.DATA_BOSTON_HOUSING
data_set_to_use = bc.DATA_SYNTHETIC_LINEAR_REGRESSION
#data_set_to_use = bc.DATA_ADIENCE_ALIGNED_CNN_1
#data_set_to_use = bc.DATA_WINE_RED

data_sets_for_exps = [data_set_to_use]

active_iterations = 2
active_items_per_iteration = 50

use_mixed_cv = False

use_baseline = False

use_pairwise = False
num_pairwise = 50
#pair_bound = (.25,1)
pair_bound = ()
use_hinge = False
noise_rate = .0
logistic_noise = 0
use_logistic_fix = True

use_bound = True
num_bound = 50
use_quartiles = True
bound_logistic = False
bound_just_constraints = True

use_neighbor = False
num_neighbor = 50
use_min_pair_neighbor = False
fast_dccp = True
init_ridge = False
init_ideal = False
init_ridge_train = False
use_neighbor_logistic = False
neighbor_convex = True

use_similar = False
num_similar = 100
use_similar_hinge = True

use_aic = True
use_test_error_for_model_selection = True



run_active_experiments = False

run_experiments = True
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


        self.use_mixed_cv = use_mixed_cv

        self.use_baseline = use_baseline

        self.use_pairwise = use_pairwise
        self.num_pairwise = num_pairwise
        self.pair_bound = pair_bound
        self.use_hinge = use_hinge
        self.noise_rate = noise_rate
        self.logistic_noise = logistic_noise

        self.use_bound = use_bound
        self.num_bound = num_bound
        self.use_quartiles = use_quartiles
        self.bound_logistic = bound_logistic
        self.bound_just_constraints = bound_just_constraints

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

        self.use_similar = use_similar
        self.num_similar = num_similar
        self.use_similar_hinge = use_similar_hinge

        self.use_test_error_for_model_selection = use_test_error_for_model_selection
        self.use_aic = use_aic

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
        elif data_set == bc.DATA_WINE_RED:
            self.set_wine_red()
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

        method_configs.use_mixed_cv = pc.use_mixed_cv
        method_configs.use_baseline = pc.use_baseline

        method_configs.use_pairwise = pc.use_pairwise
        method_configs.num_pairwise = pc.num_pairwise
        method_configs.pair_bound = pc.pair_bound
        method_configs.use_hinge = pc.use_hinge
        method_configs.noise_rate = pc.noise_rate
        method_configs.logistic_noise = pc.logistic_noise

        method_configs.use_bound = pc.use_bound
        method_configs.num_bound = pc.num_bound
        method_configs.use_quartiles = pc.use_quartiles
        method_configs.bound_logistic = pc.bound_logistic
        method_configs.bound_just_constraints = pc.bound_just_constraints

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

        method_configs.use_similar = pc.use_similar
        method_configs.num_similar = pc.num_similar
        method_configs.use_similar_hinge = pc.use_similar_hinge

        method_configs.use_test_error_for_model_selection = pc.use_test_error_for_model_selection
        method_configs.use_aic = pc.use_aic

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
            #self.learner = lap_ridge
            self.learner = relative_reg
            #self.learner = ridge_reg
            #self.learner = mean_reg

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
            base_file_name = 'RelReg-cvx-constraints-%s=%s'
            #self.files['LapRidge.pkl'] = 'Laplacian Ridge Regression'
            use_test = False
            if use_test:
                self.files['RelReg-cvx-constraints-noPairwiseReg-TEST.pkl'] = 'TEST: Ridge Regression'
            else:
                self.files['RelReg-cvx-constraints-noPairwiseReg.pkl'] = 'Ridge Regression'

            sizes = []
            #sizes.append(10)
            sizes.append(50)
            #sizes.append(100)
            #sizes.append(150)
            #sizes.append(250)
            suffixes = OrderedDict()
            #suffixes['fastDCCP'] = [None, '']
            #suffixes['init_ideal'] = [None, '']
            #suffixes['initRidgeTrain'] = ['']
            #suffixes['initRidge'] = ['']
            #suffixes['logistic'] = ['']
            #suffixes['pairBound'] = [(0,.1),(0,.25),(0,.5),(0,.75),None]
            #suffixes['pairBound'] = [(.5,1), (.25,1), None]
            #suffixes['mixedCV'] = [None,'']
            #suffixes['logNoise'] = [None, .1, .5, 1, 2]
            #suffixes['logNoise'] = [.5]
            #suffixes['logNoise'] = [None,25,50,100]
            #suffixes['baseline'] = [None,'']
            #suffixes['logFix'] = [None, '']
            suffixes['solver'] = ['SCS']
            ordered_keys = [
                'fastDCCP', 'initRidge', 'init_ideal', 'initRidgeTrain','logistic',
                'pairBound', 'mixedCV', 'logNoise',
                'baseline', 'logFix', 'solver'
            ]
            all_params = list(grid_search.ParameterGrid(suffixes))

            methods = []
            #methods.append(('numRandPairs','RelReg, %s pairs'))
            #methods.append(('numRandPairsHinge','RelReg, %s pairs hinge'))

            #methods.append(('numRandBound', 'RelReg, %s bounds'))
            #methods.append(('numRandQuartiles', 'RelReg, %s quartiles'))

            #methods.append(('numRandNeighbor', 'RelReg, %s rand neighbors'))
            #methods.append(('numMinNeighbor', 'RelReg, %s min neighbors'))

            methods.append(('numSimilar','RelReg, %s pairs'))
            methods.append(('numSimilarHinge','RelReg, %s pairs hinge'))

            for file_suffix, legend_name in methods:
                for size in sizes:
                    for params in all_params:
                        file_name = base_file_name
                        file_name = file_name % (file_suffix, str(size))
                        legend = legend_name % str(size)
                        for key in ordered_keys:
                            if not params.has_key(key):
                                continue
                            value = params[key]
                            if value is None:
                                continue
                            if value == '':
                                file_name += '-' + key
                                legend += ', ' + key
                            else:
                                file_name += '-' + key + '=' + str(value)
                                legend += ', ' + str(value) + ' ' + key
                        if use_test:
                            file_name += '-TEST'
                            legend = 'TEST: ' + legend
                        file_name += '.pkl'
                        self.files[file_name] = legend




            #self.files['RelReg-cvx-constraints-numRandNeighbor=50-fastDCCP-solver=SCS-TEST.pkl'] = 'TEST: RelReg, 50 rand neighbors, fast dccp'

            #self.files['RelReg-cvx-constraints-numRandPairs=150-mixedCV-solver=SCS.pkl'] = 'RelReg, 150 pairs, mixedCV'

            #self.files['RelReg-cvx-constraints-numRandNeighbor=50-fastDCCP-initRidge-solver=SCS.pkl'] = 'RelReg, 50 rand neighbors, init ridge'

            #self.files['RelReg-cvx-constraints-numRandNeighbor=50-initRidge-solver=SCS-TEST.pkl'] = 'TEST: RelReg, 50 rand neighbors, init ridge'
            #self.files['RelReg-cvx-constraints-numRandNeighbor=50-fastDCCP-initRidge-solver=SCS-TEST-lessParams.pkl'] = 'TEST: RelReg, 50 rand neighbors, init ridge, less params'

            #self.files['RelReg-cvx-constraints-numRandNeighbor=250-fastDCCP-initRidge-solver=SCS-TEST.pkl'] = 'TEST: RelReg, 250 rand neighbors, fast dccp, init ridge'
            #self.files['RelReg-cvx-constraints-numRandNeighbor=250-fastDCCP-initRidge-solver=SCS-TEST-lessParams.pkl'] = 'TEST: RelReg, 250 rand neighbors, fast dccp, init ridge, less params'

            '''
            self.files['RelReg-cvx-constraints-numRandPairsHinge=50-pairBound=0.75-solver=SCS-TEST.pkl'] = 'TEST: RelReg, 50 pairs hinge, .75 pair bound'
            self.files['RelReg-cvx-constraints-numRandPairsHinge=50-pairBound=0.5-solver=SCS-TEST.pkl'] = 'TEST: RelReg, 50 pairs hinge, .5 pair bound'
            self.files['RelReg-cvx-constraints-numRandPairsHinge=50-pairBound=0.25-solver=SCS-TEST.pkl'] = 'TEST: RelReg, 50 pairs hinge, .25 pair bound'
            self.files['RelReg-cvx-constraints-numRandPairsHinge=50-pairBound=0.1-solver=SCS-TEST.pkl'] = 'TEST: RelReg, 50 pairs hinge, .1 pair bound'
            '''
            '''
            self.files['RelReg-cvx-constraints-numRandPairs=50-pairBound=(0.25, 1)-solver=SCS-TEST.pkl'] = 'TEST: RelReg, 50 pairs, (.25, 1) pair bound'
            self.files['RelReg-cvx-constraints-numRandPairs=50-pairBound=(0.5, 1)-solver=SCS-TEST.pkl'] = 'TEST: RelReg, 50 pairs, (.5, 1)  pair bound'
            '''
            #self.files['RelReg-cvx-constraints-numRandPairs=50-noise=0.1-solver=SCS-TEST.pkl'] = 'TEST: RelReg, 50 pairs, .1 noise'
            '''
            self.files['RelReg-cvx-constraints-numRandPairs=50-logNoise=1-solver=SCS-TEST.pkl'] = 'TEST: RelReg, 50 pairs, 1 logistic noise'
            self.files['RelReg-cvx-constraints-numRandPairs=50-logNoise=2-solver=SCS-TEST.pkl'] = 'TEST: RelReg, 50 pairs, 2 logistic noise'
            '''
            '''
            self.files['RelReg-cvx-constraints-numRandPairs=50-pairBound=0.99-solver=SCS-TEST.pkl'] = 'TEST: RelReg, 50 pairs, .99 pair bound'
            self.files['RelReg-cvx-constraints-numRandPairs=50-pairBound=0.75-solver=SCS-TEST.pkl'] = 'TEST: RelReg, 50 pairs, .75 pair bound'
            self.files['RelReg-cvx-constraints-numRandPairs=50-pairBound=0.5-solver=SCS-TEST.pkl'] = 'TEST: RelReg, 50 pairs, .5 pair bound'
            self.files['RelReg-cvx-constraints-numRandPairs=50-pairBound=0.25-solver=SCS-TEST.pkl'] = 'TEST: RelReg, 50 pairs, .25 pair bound'
            self.files['RelReg-cvx-constraints-numRandPairs=50-pairBound=0.1-solver=SCS-TEST.pkl'] = 'TEST: RelReg, 50 pairs, .1 pair bound'
            '''
        self.figsize = (7,7)
        self.borders = (.1,.9,.9,.1)
        self.data_set_to_use = pc.data_set
        self.title = bc.data_name_dict.get(self.data_set_to_use, 'Unknown Data Set')
        self.show_legend_on_all = show_legend_on_all
        self.x_axis_string = 'Number of labeled instances'
        if pc.data_set == bc.DATA_SYNTHETIC_LINEAR_REGRESSION:
            self.ylims = [0,10]
        elif pc.data_set == bc.DATA_ADIENCE_ALIGNED_CNN_1:
            self.ylims = [0,600]


class BatchConfigs(bc.BatchConfigs):
    def __init__(self, pc):
        super(BatchConfigs, self).__init__()
        from experiment.experiment_manager import MethodExperimentManager
        self.method_experiment_manager_class = MethodExperimentManager
        run_batch = False
        if not run_batch:
            self.config_list = [MainConfigs(pc)]
            return

        c = pc.copy()
        c.use_pairwise = False
        c.use_neighbor = False
        c.use_bound = False
        c.use_hinge = False
        c.use_quartile = False
        c.use_test_error_for_model_selection = False
        self.config_list = [MainConfigs(c)]
        pairwise_params = {
            'use_pairwise': [True],
            'num_pairwise': [10, 50, 100]
        }
        self.config_list += [MainConfigs(configs) for configs in pc.generate_copies(pairwise_params)]

        bound_params = {
            'use_bound': [True],
            'num_bound': [10, 50, 100]
        }
        self.config_list += [MainConfigs(configs) for configs in pc.generate_copies(bound_params)]

        hinge_params = {
            'use_pairwise': [True],
            'use_hinge': [True],
            'num_pairwise': [10, 50, 100]
        }
        self.config_list += [MainConfigs(configs) for configs in pc.generate_copies(hinge_params)]



























