__author__ = 'Aubrey'
from collections import OrderedDict
from configs import base_configs as bc
from loss_functions import loss_function
from utility import helper_functions
from results_class import results as results_lib
from sklearn import grid_search
import numpy as np

# Command line arguments for ProjectConfigs
arguments = None

from data_sets import create_data_set

CR = []
for i in range(0,4):
    a = [create_data_set.ng_c[i],create_data_set.ng_r[i]]
    CR.append(np.asarray(a))

ST = []
for i in range(0,4):
    a = [create_data_set.ng_s[i],create_data_set.ng_t[i]]
    ST.append(np.asarray(a))

def create_project_configs():
    return ProjectConfigs()

pc_fields_to_copy = bc.pc_fields_to_copy + [
    'source_labels',
]
#data_set_to_use = bc.DATA_SYNTHETIC_LINEAR_REGRESSION
data_set_to_use = bc.DATA_NG

viz_for_paper = True

run_experiments = True

other_pc_configs = {
}

other_method_configs = {
    'include_size_in_file_name': False,
    'use_validation': False,
    'num_features': -1,
    'use_test_error_for_model_selection': False,
}

run_batch = False
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
        self.project_dir = 'hypothesis_transfer'
        self.use_pool = use_pool
        self.pool_size = pool_size
        self.source_labels = None
        self.oracle_labels = None
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
        if data_set == bc.DATA_NG:
            self.set_data_set_defaults('20ng-2000')
            self.loss_function = loss_function.ZeroOneError()
            self.cv_loss_function = loss_function.ZeroOneError()
            self.num_labels = [5, 10, 20, 40]
            self.target_labels = CR[0]
            self.source_labels = np.vstack((CR[1], ST[1]))
            self.oracle_labels = CR[1]
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
        from methods import semisupervised
        from methods import transfer_methods
        method_configs = MethodConfigs(pc)

        for key in other_method_configs.keys():
            setattr(method_configs, key, getattr(pc,key))

        target_transfer = transfer_methods.TargetTranfer(method_configs)
        target_transfer.base_learner = method.SKLRidgeClassification(method_configs)
        fuse_transfer = transfer_methods.FuseTransfer(method_configs)
        fuse_transfer.base_learner = method.SKLRidgeClassification(method_configs)
        hyp_transfer = transfer_methods.HypothesisTransfer(method_configs)

        #self.learner = fuse_transfer
        self.learner = hyp_transfer

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

        assert False, 'Not Implemented Yet'

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
        else:
            pass
            #assert False

        self.generate_file_names(pc)

    def generate_file_names(self, pc):
        self.files = OrderedDict()
        base_file_name = 'RelReg-cvx-constraints-%s=%s'
        use_test = other_method_configs['use_test_error_for_model_selection']

        self.files['TargetTransfer+SKL-RidgeClass.pkl'] = 'Target Only'
        self.files['FuseTransfer+SKL-RidgeClass.pkl'] = 'Source and Target'
        #self.files['FuseTransfer+SKL-RidgeClass-tws=0.5.pkl'] = 'Source and Target: Weighted 50%'
        #self.files['FuseTransfer+SKL-RidgeClass-tws=1.pkl'] = 'Source and Target: Weighted 100%'
        self.files['HypTransfer.pkl'] = 'Hypothesis Transfer'
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