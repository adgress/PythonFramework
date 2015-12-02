__author__ = 'Aubrey'

from configs import base_configs as bc
import numpy as np
from data_sets import create_data_set

CR = []
for i in range(0,4):
    a = [create_data_set.ng_c[i],create_data_set.ng_r[i]]
    CR.append(np.asarray(a))

class ProjectConfigs(bc.ProjectConfigs):
    def __init__(self):
        super(ProjectConfigs, self).__init__()
        self.set_boston_housing()
        self.project_dir = 'base'

    def set_ng_transfer(self):
        self.set_ng()
        self.target_labels = CR[0]
        self.source_labels = CR[1]

class MainConfigs(bc.MainConfigs):
    def __init__(self):
        super(MainConfigs, self).__init__()


class VisualizationConfigs(bc.VisualizationConfigs):
    def __init__(self):
        super(VisualizationConfigs, self).__init__()

class BatchConfigs(bc.BatchConfigs):
    def __init__(self):
        super(BatchConfigs, self).__init__()
        self.config_list = [MainConfigs()]