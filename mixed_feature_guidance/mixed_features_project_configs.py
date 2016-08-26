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
]
data_set_to_use = bc.DATA_SYNTHETIC_LINEAR_REGRESSION_10_nnz4

viz_for_paper = False

run_experiments = True

other_pc_configs = {
}

other_method_configs = {
    'include_size_in_file_name': False,
    'num_features': -1,
    'use_test_error_for_model_selection': True,
    'y_scale_min_max': False,
    'y_scale_standard': False,
    'scipy_opt_method': 'L-BFGS-B',
    'num_cv_splits': 10,
    'eps': 1e-10,
    'use_perfect_feature_selection': True
}

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
            bc.apply_arguments(self, arguments)

        for key, value in other_method_configs.items():
            setattr(self, key, value)


    def set_data_set(self, data_set):
        self.data_set = data_set
        if data_set == bc.DATA_BOSTON_HOUSING:
            self.set_data_set_defaults('boston_housing')
            self.num_labels = [5, 10, 20, 40]
        elif data_set == bc.DATA_SYNTHETIC_LINEAR_REGRESSION_10_nnz4:
            self.set_data_set_defaults('synthetic_linear_reg500-10-1-nnz=4')
            self.num_labels = [10, 20, 40]
        elif data_set == bc.DATA_ADIENCE_ALIGNED_CNN_1:
            self.set_data_set_defaults('adience_aligned_cnn_1_per_instance_id')
            self.num_labels = [10, 20, 40]
        elif data_set == bc.DATA_WINE_RED:
            self.set_data_set_defaults('wine-red')
            self.num_labels = [5, 10, 20, 40]
        elif data_set == bc.DATA_CONCRETE:
            self.set_data_set_defaults('concrete')
            self.num_labels = [5, 10, 20, 40]
        elif data_set == bc.DATA_DROSOPHILIA:
            self.set_data_set_defaults('drosophila')
            self.num_labels = [10,20,40]
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
        from methods import mixed_feature_guidance
        method_configs = MethodConfigs(pc)

        for key in other_method_configs.keys():
            setattr(method_configs, key, getattr(pc,key))

        ridge = method.SKLRidgeRegression(method_configs)

        #self.learner = ssl_reg
        #self.learner = nw_reg
        self.learner = mixed_feature_guidance.MixedFeatureGuidanceMethod(method_configs)
        #self.learner = ridge

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
        else
            self.config_list = [MainConfigs(pc)]
            #assert False, 'Not Implemented Yet'

class VisualizationConfigs(bc.VisualizationConfigs):
    def __init__(self, data_set=None, **kwargs):
        super(VisualizationConfigs, self).__init__(data_set, **kwargs)
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

        self.generate_file_names(pc)

    def generate_file_names(self, pc):
        self.files = OrderedDict()
        base_file_name = 'RelReg-cvx-constraints-%s=%s'
        use_test = other_method_configs['use_test_error_for_model_selection']

        #self.files['SLL-NW.pkl'] = 'LLGC'
        #self.files['NW.pkl'] = 'NW'
        self.files['SKL-RidgeReg.pkl'] = 'SKL Ridge Regression'
        self.files['Mixed-feats_method=Ridge.pkl'] = 'Mixed: Ridge'
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


viz_params = [
    {'None': None},
]