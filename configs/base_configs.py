from loss_functions import loss_function
from results_class import results as results_lib
from copy import deepcopy
from sklearn import grid_search
import numpy as np

DATA_NG = 1
DATA_BOSTON_HOUSING = 2
DATA_CONCRETE = 3
DATA_BIKE_SHARING = 4
DATA_WINE = 5
DATA_ADIENCE_ALIGNED_CNN_1 = 6
DATA_WINE_RED = 7
DATA_DROSOPHILIA = 8
DATA_KC_HOUSING = 9
DATA_POLLUTION_2 = 10
DATA_CLIMATE_MONTH = 11
DATA_UBER = 12
DATA_IRS = 13
DATA_DS2 = 14
DATA_TAXI = 15
DATA_ZILLOW = 16

DATA_PAIR_START = 100
DATA_PAIR_82_83 = 101
DATA_PAIR_13_14 = 102
DATA_PAIR_END = 103

DATA_SYNTHETIC_START = 1000
DATA_SYNTHETIC_CLASSIFICATION_LOCAL = DATA_SYNTHETIC_START + 1
DATA_SYNTHETIC_STEP_TRANSFER = DATA_SYNTHETIC_START + 2
DATA_SYNTHETIC_STEP_LINEAR_TRANSFER = DATA_SYNTHETIC_START + 3
DATA_SYNTHETIC_CLASSIFICATION = DATA_SYNTHETIC_START + 4
DATA_SYNTHETIC_DELTA_LINEAR = DATA_SYNTHETIC_START + 5
DATA_SYNTHETIC_CROSS = DATA_SYNTHETIC_START + 6
DATA_SYNTHETIC_SLANT = DATA_SYNTHETIC_START + 7
DATA_SYNTHETIC_CURVE = DATA_SYNTHETIC_START + 8
DATA_SYNTHETIC_FLIP = DATA_SYNTHETIC_START + 9

DATA_SYNTHETIC_HYP_TRANS_1_1 = DATA_SYNTHETIC_START + 9
DATA_SYNTHETIC_HYP_TRANS_2_2 = DATA_SYNTHETIC_START + 10

DATA_SYNTHETIC_PIECEWISE = DATA_SYNTHETIC_START + 11
DATA_SYNTHETIC_SLANT_MULTITASK = DATA_SYNTHETIC_START + 12

DATA_SYNTHETIC_LINEAR_REGRESSION = 2000
DATA_SYNTHETIC_LINEAR_REGRESSION_10_nnz4 = 2001
DATA_SYNTHETIC_LINEAR_REGRESSION_10 = 2002

data_name_dict = {
    DATA_NG: '20ng',
    DATA_BOSTON_HOUSING: 'Boston Housing',
    DATA_SYNTHETIC_STEP_TRANSFER: 'Synthetic Step',
    DATA_SYNTHETIC_STEP_LINEAR_TRANSFER: 'Synthetic Step',
    DATA_SYNTHETIC_CLASSIFICATION: 'Synthetic Classification',
    DATA_CONCRETE: 'Concrete',
    DATA_BIKE_SHARING: 'Bike Sharing',
    DATA_WINE: 'Wine',
    DATA_SYNTHETIC_DELTA_LINEAR: 'Synthetic Delta',
    DATA_SYNTHETIC_CROSS: 'Synthetic Cross',
    DATA_SYNTHETIC_SLANT: 'Synthetic Slant',
    DATA_SYNTHETIC_CURVE: 'Synthetic Curve',
    DATA_SYNTHETIC_LINEAR_REGRESSION: 'Synthetic Linear Regression',
    DATA_SYNTHETIC_LINEAR_REGRESSION_10_nnz4: 'Synthetic Linear Regression p=10, nnz=4',
    DATA_SYNTHETIC_PIECEWISE: 'Synthetic Piecewise',
    DATA_SYNTHETIC_SLANT_MULTITASK: 'Synthetic Slant Multitask',
    DATA_ADIENCE_ALIGNED_CNN_1: 'Adience Aligned CNN 1 Per Instance ID',
    DATA_WINE_RED: 'Wine: Red',
    DATA_DROSOPHILIA: 'Drosophila',
    DATA_DS2: 'ITS',
    DATA_TAXI: 'Taxi',
    DATA_IRS: 'Census',
    DATA_CLIMATE_MONTH: 'Temperature',
    DATA_SYNTHETIC_PIECEWISE: 'Synthetic Piecewise',
    DATA_ZILLOW: 'Taxi+Housing',
    DATA_KC_HOUSING: 'King County Housing',
    DATA_SYNTHETIC_LINEAR_REGRESSION_10: 'Synthetic Linear'
}

def is_synthetic_data(data_idx):
    return data_idx > DATA_SYNTHETIC_START

def is_pair_data(data_idx):
    return data_idx > DATA_PAIR_START and data_idx < DATA_PAIR_END

def apply_arguments(configs, arguments):
    if arguments.num_labels is not None:
        configs.overwrite_num_labels = arguments.num_labels
    if arguments.split_idx is not None:
        configs.split_idx = arguments.split_idx

class Configs(object):
    def __init__(self, **kwargs):
        for key,value in kwargs.items():
            setattr(self, key, value)
        self.overwrite_num_labels = None
        self.split_idx = None
        pass

    def copy(self, key=None, value=None):
        c2 = deepcopy(self)
        if key is not None:
            setattr(c2, key, value)
        return c2

    def generate_copies(self, dict):
        copies = []
        param_grid = list(grid_search.ParameterGrid(dict))
        for params in param_grid:
            c2 = deepcopy(self)
            c2.set(**params)
            copies.append(c2)
        return copies

    def set(self, **kwargs):
        for key, value in kwargs.iteritems():
            setattr(self, key, value)

    def has(self,key):
        return hasattr(self,key)

    def get(self,key,default_value):
        if not self.has(key):
            return default_value
        return getattr(self, key)

    def stringify_fields(self,fields):
        assert False

    def copy_fields(self,other,keys):
        for k in keys:
            setattr(self,k,getattr(other,k))

    @property
    def data_file(self):
        pc = create_project_configs()
        s = self.data_dir + '/' + self.data_set_file_name
        return s

    @property
    def results_directory(self):
        s = self.project_dir + '/' + self.results_dir
        return s

    def set_boston_housing(self):
        self.data_dir = 'data_sets/boston_housing'
        self.data_name = 'boston_housing'
        self.data_set_file_name = 'split_data.pkl'
        self.results_dir = 'boston_housing'

    @property
    def num_labels(self):
        if self.overwrite_num_labels is not None:
            return [self.overwrite_num_labels]
        return self._num_labels

    @num_labels.setter
    def num_labels(self, value):
        self._num_labels = value

    @property
    def should_load_temp_data(self):
        return self.overwrite_num_labels is None and self.split_idx is None


def create_project_configs():
    return ProjectConfigs()

class ProjectConfigs(Configs):
    def __init__(self, data_set=None):
        super(ProjectConfigs, self).__init__()
        self._num_labels = None
        self.data_set = data_set
        self.project_dir = 'base'
        self.loss_function = loss_function.MeanSquaredError()
        self.cv_loss_function = loss_function.MeanSquaredError()
        self.data_dir = ''
        self.data_name = ''
        self.data_set_file_name = ''
        self.results_dir = ''
        self.include_name_in_results = False
        self.labels_to_use = None
        self.labels_to_not_sample = None
        self.target_labels = None
        self.source_labels = None
        self.oracle_labels = None
        self.num_labels = range(40,201,40)
        #self.num_labels = range(40,81,40)
        self.set_boston_housing()
        self.num_splits = 30
        self.labels_to_keep = None
        self.labels_to_not_sample = {}
        self.data_set = None
        self.use_pool = False
        self.pool_size = 2
        self.method_results_class = results_lib.MethodResults

        self.oracle_data_set_ids = None

    def set_data_set(self, data_set):
        assert False

    def set_data_set_defaults(self, data_set_name, target_labels=None, source_labels=None, is_regression=True):
        assert is_regression
        self.loss_function = loss_function.MeanSquaredError()
        self.cv_loss_function = loss_function.MeanSquaredError()
        self.data_dir = 'data_sets/' + data_set_name
        self.data_name = data_set_name
        self.results_dir = data_set_name
        self.data_set_file_name = 'split_data.pkl'
        self.target_labels = target_labels
        self.source_labels = source_labels
        if target_labels is not None:
            self.target_labels = np.asarray(target_labels)
        if source_labels is not None:
            self.source_labels = np.asarray(source_labels)

    def set_data_set(self, name, target_labels, source_labels, is_regression):
        assert is_regression
        self.loss_function = loss_function.MeanSquaredError()
        self.cv_loss_function = loss_function.MeanSquaredError()

        self.data_dir = 'data_sets/' + name
        self.data_name = name
        self.results_dir = name

        self.data_set_file_name = 'split_data.pkl'
        self.target_labels = np.asarray(target_labels)
        self.source_labels = np.asarray(source_labels)



pc_fields_to_copy = ['data_dir',
                     'data_name',
                     'data_set_file_name',
                     'project_dir',
                     'results_dir',
                     'include_name_in_results',
                     'labels_to_use',
                     'num_labels',
                     'num_splits',
                     'loss_function',
                     'cv_loss_function',
                     'labels_to_keep',
                     'labels_to_not_sample',
                     'target_labels',
                     'use_pool',
                     'pool_size',
                     'data_set',
                     'overwrite_num_labels',
                     'split_idx',
                     'method_results_class']

class BatchConfigs(Configs):
    def __init__(self):
        super(BatchConfigs, self).__init__()
        self.config_list = [MainConfigs()]

class MainConfigs(Configs):
    def __init__(self):
        super(MainConfigs, self).__init__()
        pc = create_project_configs()
        self.copy_fields(pc,pc_fields_to_copy)
        self.learner = None
        self.set_ridge_regression()

    @property
    def results_file(self):
        return self.results_directory + '/' + self.learner.name_string + '.pkl'

    def set_ridge_regression(self):
        from methods import method
        self.learner = method.SKLRidgeRegression(MethodConfigs())

    def set_l2_logistic_regression(self):
        from methods import method
        self.learner = method.SKLLogisticRegression()

    def set_l2_svm(self):
        assert False

    def set_mean(self):
        from methods import method
        self.learner = method.SKLGuessClassifier()

    def set_median(self):
        assert False

    def set_nadaraya_watson(self):
        assert False


class MethodConfigs(Configs):
    def __init__(self):
        super(MethodConfigs, self).__init__()
        pc = create_project_configs()
        self.z_score = False
        self.quiet = False
        self.loss_function = loss_function.MeanSquaredError()
        self.cv_loss_function = loss_function.MeanSquaredError()
        self.cv_loss_function = pc.cv_loss_function
        self.loss_function = pc.loss_function
        self.use_validation = False
        self.metric = 'euclidean'

    @property
    def results_file_name(self):
        assert False

class DataProcessingConfigs(Configs):
    def __init__(self):
        pc = create_project_configs()
        self.copy_fields(pc,pc_fields_to_copy)
        self.splits = 30
        self.is_regression = False
        self.labels_to_use = None
        self.split_data_set_ids = None
        self.data_set_ids_to_keep = None

class VisualizationConfigs(Configs):
    def __init__(self, data_set=None, **kwargs):
        super(VisualizationConfigs, self).__init__(**kwargs)
        pc = ProjectConfigs(data_set)
        self.copy_fields(pc,pc_fields_to_copy)
        self.axis_to_use = None
        self.show_legend = True
        self.sizes_to_use = None
        self.x_axis_field = 'num_labels'
        self.x_axis_string = 'Size'
        self.y_axis_string = 'Error'
        self.figsize = None
        self.borders = (.1,.9,.9,.1)
        self.fontsize = None
        self.data_set_to_use = pc.data_set
        self.show_legend_on_all = True
        self.max_rows = 3
        self.vis_table = False
        self.size_to_vis = None
        self.sizes_to_use = None
        self.title = 'Title not set'
        self.baseline_idx = None

        self.files = [
            'SKL-RidgeReg.pkl'
        ]

    def _results_files(self):
        dir = self.results_directory
        files = []
        for key, value in self.files.iteritems():
            files.append((dir + '/' + key, value))
        return files

    @property
    def results_files(self):
        return self._results_files()


if __name__ == "__main__":
    c = Configs()