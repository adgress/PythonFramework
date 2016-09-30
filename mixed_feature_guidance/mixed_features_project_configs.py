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
# Command line arguments for ProjectConfigs
arguments = None

def create_project_configs():
    return ProjectConfigs()

pc_fields_to_copy = bc.pc_fields_to_copy + [
]
#data_set_to_use = bc.DATA_SYNTHETIC_LINEAR_REGRESSION_10_nnz4
#data_set_to_use = bc.DATA_SYNTHETIC_LINEAR_REGRESSION
data_set_to_use = bc.DATA_DROSOPHILIA
#data_set_to_use = bc.DATA_BOSTON_HOUSING
#data_set_to_use = bc.DATA_WINE_RED
#data_set_to_use = bc.DATA_DROSOPHILIA
#data_set_to_use = bc.DATA_CONCRETE
#data_set_to_use = bc.DATA_KC_HOUSING

viz_for_paper = False

run_experiments = True

use_ridge = False
use_mean = False
use_quad_feats = False
#mixed_feature_method = mixed_feature_guidance.MixedFeatureGuidanceMethod.METHOD_RELATIVE
#mixed_feature_method = mixed_feature_guidance.MixedFeatureGuidanceMethod.METHOD_HARD_CONSTRAINT
mixed_feature_method = mixed_feature_guidance.MixedFeatureGuidanceMethod.METHOD_RIDGE

viz_w_error = False

other_pc_configs = {
}

other_method_configs = {
    'num_random_pairs': 10,
    'num_random_signs': 0,
    'num_features': None,
    'use_nonneg': False,
    'use_stacking': False,
    'use_corr': True,
    'include_size_in_file_name': False,
    'num_features': -1,
    'use_test_error_for_model_selection': False,
    'y_scale_min_max': False,
    'y_scale_standard': False,
    'scipy_opt_method': 'L-BFGS-B',
    'cvx_method': 'CVXOPT',
    'num_cv_splits': 10,
    'eps': 1e-10,
    'use_perfect_feature_selection': True
}

if data_set_to_use == bc.DATA_DROSOPHILIA:
    other_method_configs['num_features'] = 50

run_batch = True
if helper_functions.is_laptop():
    run_batch = False

show_legend_on_all = True

max_rows = 3

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
        self.project_dir = 'mixed_feature_guidance'
        self.use_pool = use_pool
        self.pool_size = pool_size
        if data_set is None:
            data_set = data_set_to_use
        self.set_data_set(data_set)
        self.num_splits = 30
        if use_arguments and arguments is not None:
            apply_arguments(self)

        for key, value in other_method_configs.items():
            setattr(self, key, value)


    def set_data_set(self, data_set):
        self.data_set = data_set
        if data_set == bc.DATA_BOSTON_HOUSING:
            self.set_data_set_defaults('boston_housing')
            self.num_labels = [10, 20, 40, 80]
            #self.num_labels = [80]
        elif data_set == bc.DATA_SYNTHETIC_LINEAR_REGRESSION:
            self.set_data_set_defaults('synthetic_linear_reg500-50-1.01')
            self.num_labels = [5, 10, 20, 40]
        elif data_set == bc.DATA_SYNTHETIC_LINEAR_REGRESSION_10_nnz4:
            self.set_data_set_defaults('synthetic_linear_reg500-10-1-nnz=4')
            self.num_labels = [10, 20, 40]
            #self.num_labels = [10]
        elif data_set == bc.DATA_ADIENCE_ALIGNED_CNN_1:
            self.set_data_set_defaults('adience_aligned_cnn_1_per_instance_id')
            self.num_labels = [10, 20, 40]
        elif data_set == bc.DATA_WINE_RED:
            self.set_data_set_defaults('wine-red')
            self.num_labels = [10, 20, 40]
            #self.num_labels = [160]
        elif data_set == bc.DATA_CONCRETE:
            self.set_data_set_defaults('concrete')
            self.num_labels = [5, 10, 20, 40]
        elif data_set == bc.DATA_DROSOPHILIA:
            self.set_data_set_defaults('drosophilia')
            self.num_labels = [10,20,40]
            #self.num_labels = [10, 20, 40, 80, 160]
        elif data_set == bc.DATA_KC_HOUSING:
            self.set_data_set_defaults('kc_housing')
            self.num_labels = [10, 20, 40]
            #self.num_labels = [160]
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
        from methods import active_methods
        from methods import semisupervised
        from methods import preprocessing
        from methods import mixed_feature_guidance
        method_configs = MethodConfigs(pc)

        for key in other_method_configs.keys():
            setattr(method_configs, key, getattr(pc,key))

        ridge = method.SKLRidgeRegression(method_configs)
        mean_estimator = method.SKLMeanRegressor(method_configs)
        if use_quad_feats:
            ridge.preprocessor = preprocessing.BasisQuadraticPreprocessor()
            #ridge.preprocessor = preprocessing.BasisQuadraticFewPreprocessor()
        if use_mean:
            self.learner = mean_estimator
        elif use_ridge:
            self.learner = ridge
        else:
            method_configs.method = getattr(pc, 'mixed_feature_method', mixed_feature_method)
            self.learner = mixed_feature_guidance.MixedFeatureGuidanceMethod(method_configs)


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
        else:
            #self.config_list = [MainConfigs(pc)]
            self.config_list = []
            p = deepcopy(pc)
            p.mixed_feature_method = mixed_feature_guidance.MixedFeatureGuidanceMethod.METHOD_RELATIVE
            p.num_random_pairs = 0
            p.num_random_signs = 10
            self.config_list.append(MainConfigs(p))
            p = deepcopy(p)
            p.num_random_pairs = 10
            p.num_random_signs = 0
            self.config_list.append(MainConfigs(p))
            p = deepcopy(pc)
            p.mixed_feature_method = mixed_feature_guidance.MixedFeatureGuidanceMethod.METHOD_RIDGE
            self.config_list.append(MainConfigs(p))
            #assert False, 'Not Implemented Yet'

class VisualizationConfigs(bc.VisualizationConfigs):
    def __init__(self, data_set=None, **kwargs):
        super(VisualizationConfigs, self).__init__(data_set, **kwargs)
        self.max_rows = 2
        pc = ProjectConfigs(data_set)
        self.copy_fields(pc,pc_fields_to_copy)
        if viz_w_error:
            self.loss_function = loss_function.LossFunctionParamsMeanAbsoluteError()

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
            self.ylims = [0,100]
        elif pc.data_set == bc.DATA_CONCRETE:
            self.ylims = [0,1000]
        elif pc.data_set == bc.DATA_DROSOPHILIA:
            self.ylims = [0,3]

        self.generate_file_names(pc)

    def generate_file_names(self, pc):
        self.files = OrderedDict()
        base_file_name = 'RelReg-cvx-constraints-%s=%s'
        use_test = other_method_configs['use_test_error_for_model_selection']

        #self.files['Mixed-feats_QuadFeatsFew_method=HardConstraints_signs=20.pkl'] = 'Mixed: Hard Constraints, Quad Feats Few, 20 signs'
        #self.files['SKL-RidgeReg-QuadFeatsFew.pkl'] = 'SKL Ridge Regression, Quad Feats Few'
        #self.files['SLL-NW.pkl'] = 'LLGC'
        #self.files['NW.pkl'] = 'NW'

        if other_method_configs['use_nonneg']:
            self.files['Mixed-feats_method=Ridge_nonneg.pkl'] = 'Mixed: Ridge, nonneg'
            self.files['Mixed-feats_method=Rel_pairs=10_corr_nonneg.pkl'] = 'Mixed: Relative, corr, 10 pairs, nonneg'
            self.files['Mixed-feats_method=Rel_pairs=1000_corr_nonneg.pkl'] = 'Mixed: Relative, corr, 1000 pairs, nonneg'
            self.files['Mixed-feats_method=Ridge_nonneg_stacked.pkl'] = 'Mixed: Ridge, nonneg, stacked'
            self.files['Mixed-feats_method=Rel_pairs=10_corr_nonneg_stacked.pkl'] = 'Mixed: Relative, corr, nonneg, stacked, 10 pairs'
        else:
            '''
            self.files['SKL-RidgeReg.pkl'] = 'SKL Ridge Regression'
            self.files['Mixed-feats_method=Ridge.pkl'] = 'Mixed: Ridge'
            self.files['Mixed-feats_method=Rel_signs=10_corr_CVXOPT.pkl'] = 'Mixed: Relative, corr, 10 signs'
            self.files['Mixed-feats_method=Rel_pairs=10_corr_CVXOPT.pkl'] = 'Mixed: Relative, corr, 10 pairs'
            self.files['Mixed-feats_method=Rel_signs=1000_corr_CVXOPT.pkl'] = 'Mixed: Relative, corr, 1000 signs'
            '''
            self.files['Mixed-feats_method=Ridge_CVXOPT.pkl'] = 'Mixed: Ridge'
            self.files['Mixed-feats_method=Rel_signs=10_corr_CVXOPT.pkl'] = 'Mixed: Relative, corr, 10 signs'
            self.files['Mixed-feats_method=Rel_signs=50_corr_CVXOPT.pkl'] = 'Mixed: Relative, corr, 50 signs'
            #self.files['Mixed-feats_method=Ridge_stacked.pkl'] = 'Mixed: Ridge, stacked'
            self.files['Mixed-feats_method=Rel_pairs=10_corr_CVXOPT.pkl'] = 'Mixed: Relative, corr, 10 pairs'
            #self.files['Mixed-feats_method=Rel_signs=10_corr_stacked.pkl'] = 'Mixed: Relative, corr, stacked, 10 signs'
        '''
        self.files['Mixed-feats_method=Ridge.pkl'] = 'Mixed: Ridge'
        self.files['Mixed-feats_method=HardConstraints_pairs=5.pkl'] = 'Mixed: Hard Constraints, 5 pairs'
        self.files['Mixed-feats_method=HardConstraints_pairs=10.pkl'] = 'Mixed: Hard Constraints, 10 pairs'
        self.files['Mixed-feats_method=HardConstraints_pairs=20.pkl'] = 'Mixed: Hard Constraints, 20 pairs'
        self.files['Mixed-feats_method=HardConstraints_signs=5.pkl'] = 'Mixed: Hard Constraints, 5 signs'
        self.files['Mixed-feats_method=HardConstraints_signs=10.pkl'] = 'Mixed: Hard Constraints, 10 signs'
        self.files['Mixed-feats_method=HardConstraints_signs=25.pkl'] = 'Mixed: Hard Constraints, 25 signs'
        '''
        #self.files['SKL-DumReg.pkl'] = 'Predict Mean'
        sizes = []
        suffixes = OrderedDict()
        #suffixes['mixedCV'] = [None,'']
        if not use_test:
            suffixes['nCV'] = [None, '10']

        #suffixes['numFeats'] = [str(num_feat)]

        ordered_keys = [
            'nCV',
        ]

        if use_test:
            test_files = {}
            for f, leg in self.files.iteritems():
                f = helper_functions.remove_suffix(f, '.pkl')
                f += '-TEST.pkl'
                leg = 'TEST: ' + leg
                test_files[f] = leg
            self.files = test_files

        methods = []
        #methods.append(('numRandPairs','RelReg, %s pairs', 'Our Method: %s relative'))
        self.title = 'Test'

        all_params = list(grid_search.ParameterGrid(suffixes))
        for file_suffix, legend_name, legend_name_paper in methods:
            for size in sizes:
                for params in all_params:
                    file_name = base_file_name
                    file_name = file_name % (file_suffix, str(size))
                    legend = legend_name
                    if viz_for_paper:
                        legend = legend_name_paper
                    legend %= str(size)
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
        self.files['SKL-DumReg.pkl'] = 'Predict Mean'



viz_params = [
    {'None': None},
]