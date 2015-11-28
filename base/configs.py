__author__ = 'Aubrey'

from configs import base_configs as bc

class ProjectConfigs(bc.ProjectConfigs):
    def __init__(self):
        super(bc.ProjectConfigs, self).__init__()

class MainConfigs(bc.MainConfigs):
    def __init__(self):
        super(bc.MainConfigs, self).__init__()

class VisualizationConfigs(bc.VisualizationConfigs):
    def __init__(self):
        super(bc.VisualizationConfigs, self).__init__()
