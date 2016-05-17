from loss_functions import loss_function
from results_class import results as results_lib

DATA_NG = 1
DATA_BOSTON_HOUSING = 2
DATA_CONCRETE = 3
DATA_BIKE_SHARING = 4
DATA_WINE = 5
DATA_ADIENCE_ALIGNED_CNN_1 = 6
DATA_WINE_RED = 7

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

DATA_SYNTHETIC_LINEAR_REGRESSION = 2000

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
    DATA_ADIENCE_ALIGNED_CNN_1: 'Adience Aligned CNN 1 Per Instance ID',
    DATA_WINE_RED: 'Wine: Red',
}

def is_synthetic_data(data_idx):
    return data_idx > DATA_SYNTHETIC_START

def is_pair_data(data_idx):
    return data_idx > DATA_PAIR_START and data_idx < DATA_PAIR_END

class Configs(object):
    def __init__(self):
        self.overwrite_num_labels = None
        self.split_idx = None
        pass

    def has(self,key):
        return hasattr(self,key)

    def get(self,key,default_value):
        if ~self.has(key):
            assert default_value in locals()
            return default_value
        return getattr(key)

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

    def set_data_set(self, data_set):
        assert False




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

class VisualizationConfigs(Configs):
    def __init__(self, data_set=None):
        super(VisualizationConfigs, self).__init__()
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