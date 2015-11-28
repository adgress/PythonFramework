from loss_functions import loss_function

class Configs(object):
    def __init__(self):
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
        assert False


    def set_uci_yeast(self):
        self.data_dir = 'data_sets/uci_yeast'
        self.data_name = 'uci_yeast'
        self.data_set_file_name = ''
        self.results_dir = 'uci_yeast'

def create_project_configs():
    return ProjectConfigs()

class ProjectConfigs(Configs):
    def __init__(self):
        self.project_dir = 'base'
        self.loss_function = loss_function.MeanSquaredError()
        self.data_dir = ''
        self.data_name = ''
        self.data_set_file_name = ''
        self.results_dir = ''
        self.include_name_in_results = False
        self.labels_to_use = None
        self.set_uci_yeast()


pc_fields_to_copy = ['data_dir',
                     'data_name',
                     'data_set_file_name',
                     'results_dir',
                     'include_name_in_results',
                     'labels_to_use']

class MainConfigs(Configs):
    def __init__(self):
        pc = create_project_configs()
        self.copy_fields(pc,pc_fields_to_copy)
        pass

    def set_ridge_regression(self):
        pass

    def set_l2_logistic_regression(self):
        pass

    def set_l2_svm(self):
        pass

    def set_mean(self):
        pass

    def set_median(self):
        pass

    def set_nadaraya_watson(self):
        pass



    @property
    def results_directory(self):
        pc = create_project_configs()
        s = pc.project_dir + '/' + self.data_name + '/'
        if self.include_name_in_results:
            s += self.data_set_name + '/'
        return s


class MethodConfigs(Configs):
    def __init__(self):
        self.z_score = False
        self.quiet = False
        pass

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
    def __init__(self):
        pc = create_project_configs()
        self.copy_fields(pc,pc_fields_to_copy)
        self.axis_to_use = None
        self.show_legend = True
        self.size_field = ''
        self.sizes_to_use = None
        self.x_axis_field = None
        self.x_axis_string = 'Size'
        self.y_axis_string = 'Error'
        pass




if __name__ == "__main__":
    c = Configs()