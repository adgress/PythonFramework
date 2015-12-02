__author__ = 'Aubrey'

from configs import base_configs as bc

class ProjectConfigs(bc.ProjectConfigs):
    def __init__(self):
        super(ProjectConfigs, self).__init__()
        self.set_boston_housing()
        self.project_dir = 'base'

class MainConfigs(bc.MainConfigs):
    def __init__(self):
        super(MainConfigs, self).__init__()

class VisualizationConfigs(bc.VisualizationConfigs):
    def __init__(self):
        super(VisualizationConfigs, self).__init__()

class BatchConfigs(bc.BatchConfigs):
    def __init__(self):
        super(BatchConfigs, self).__init__()