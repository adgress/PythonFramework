__author__ = 'Aubrey'

import abc
from utility import helper_functions
from results_class import results
import os
import numpy as np
import math
from utility import array_functions
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

    def _temp_split_file_name(self, final_file_name, split):
        directory = helper_functions.remove_suffix(final_file_name, '.pkl')
        return directory + '/split=' + str(split) + '.pkl'

    def _temp_experiment_file_name(self, final_file_name, num_labels):
        directory = helper_functions.remove_suffix(final_file_name, '.pkl')
        return directory + '/num_labels=' + str(num_labels) + '.pkl'

    def _load_temp_split_file(self, final_file_name, split):
        split_temp_file = self._temp_split_file_name(final_file_name, split)
        if not os.path.isfile(split_temp_file):
            return None
        print 'found ' + split_temp_file + ' - loading'
        return helper_functions.load_object(split_temp_file)

    def _load_temp_experiment_file(self, final_file_name, num_labels):
        experiment_temp_file = self._temp_experiment_file_name(final_file_name, num_labels)
        if not os.path.isfile(experiment_temp_file):
            return None
        print 'found ' + experiment_temp_file + ' - loading'
        return helper_functions.load_object(experiment_temp_file)

    def _delete_temp_split_files(self, file, num_splits):
        for i in range(num_splits):
            helper_functions.delete_file(self._temp_split_file_name(file, i))

    def _delete_temp_experiment_files(self, file, num_labels):
        for i in num_labels:
            helper_functions.delete_file(self._temp_experiment_file_name(file, i))

    def _delete_temp_folder(self, file):
        folder = helper_functions.remove_suffix(file, '.pkl')
        helper_functions.delete_dir_if_empty(folder)

    def run_experiments(self):
        data_file = self.configs.data_file
        data_and_splits = helper_functions.load_object(data_file)
        data_and_splits.data.repair_data()
        assert self.configs.num_splits <= len(data_and_splits.splits)
        data_and_splits.labels_to_keep = self.configs.labels_to_keep
        data_and_splits.labels_to_not_sample = self.configs.labels_to_not_sample
        data_and_splits.data.repair_data()
        method_results = results.MethodResults()
        results_file = self.configs.results_file
        if os.path.isfile(results_file):
            print results_file + ' already exists - skipping'
            return
        for i,num_labels in enumerate(self.configs.num_labels):
            experiment_results = self._load_temp_experiment_file(results_file,num_labels)
            if not experiment_results:
                print 'num_labels: ' + str(num_labels)
                experiment_results = results.ExperimentResults()

                for split in range(self.configs.num_splits):
                    curr_results = self._load_temp_split_file(results_file, split)
                    if not curr_results:
                        print 'split: ' + str(split)
                        curr_data = data_and_splits.get_split(split, num_labels)
                        curr_results = self.configs.learner.train_and_test(curr_data)
                        helper_functions.save_object(self._temp_split_file_name(results_file,split),curr_results)
                    experiment_results.append(curr_results)
                helper_functions.save_object(self._temp_experiment_file_name(results_file,num_labels),experiment_results)
            experiment_results.num_labels = num_labels
            method_results.append(experiment_results)
            self._delete_temp_split_files(results_file, self.configs.num_splits)

            print 'Mean Error:' + str(method_results.compute_error(self.configs.loss_function)[i].mean)
        #a = method_results.compute_error(self.configs.loss_function)
        method_results.configs = self.configs
        helper_functions.save_object(results_file,method_results)
        self._delete_temp_experiment_files(results_file, self.configs.num_labels)
        self._delete_temp_folder(results_file)
        pass
