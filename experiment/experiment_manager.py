__author__ = 'Aubrey'

import abc
from utility import helper_functions
from results_class import results

class ExperimentManager(object):
    __metaclass__ = abc.ABCMeta
    def __init__(self, configs=None):
        self.configs = configs

    @abc.abstractmethod
    def run_experiments(self):
        pass


class BatchExperimentManager(ExperimentManager):
    def __init__(self,configs=None):
        super(BatchExperimentManager, self).__init__(configs)
        pass

    def run_experiments(self):
        for configs in self.configs.config_list:
            exp_manager = MethodExperimentManager(configs)
            exp_manager.run_experiments()

class MethodExperimentManager(ExperimentManager):
    def __init__(self,configs=None):
        super(MethodExperimentManager,self).__init__(configs)
        pass

    def run_experiments(self):
        data_file = self.configs.data_file
        data_and_splits = helper_functions.load_object(data_file)
        assert self.configs.num_splits <= len(data_and_splits.splits)
        method_results = results.MethodResults()
        for num_labels in self.configs.num_labels:
            experiment_results = results.ExperimentResults()
            for split in range(self.configs.num_splits):
                curr_data = data_and_splits.get_split(split, num_labels)
                curr_results = self.configs.learner.train_and_test(curr_data)
                experiment_results.append(curr_results)
            experiment_results.num_labels = num_labels
            method_results.append(experiment_results)
        a = method_results.compute_error(self.configs.loss_function)
        results_file = self.configs.results_file
        method_results.configs = self.configs
        helper_functions.save_object(results_file,method_results)
        pass
