__author__ = 'Aubrey'

import abc

class ExperimentManager:
    __metaclass__ = abc.ABCMeta
    def __init__(self, configs=None):
        self.configs = configs

    @abc.abstractmethod
    def run_experiment(self):
        pass


class BatchExperimentManage(ExperimentManager):
    def __init__(self,configs=None):
        super(ExperimentManager,self).__init__(configs)
        pass

class MethodExperimentManager(ExperimentManager):
    def __init__(self,configs=None):
        super(ExperimentManager,self).__init__(configs)
        pass