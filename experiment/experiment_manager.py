__author__ = 'Aubrey'

import abc
from utility import helper_functions
from utility import multiprocessing_utility
from results_class import results
import os
import numpy as np
import math
from utility import array_functions
import multiprocessing
import itertools
import copy
from utility import helper_functions
from utility import mpi_utility

def _temp_split_file_name(final_file_name, num_labels, split):
    directory = helper_functions.remove_suffix(final_file_name, '.pkl')
    return directory + '/num_labels=' + str(num_labels) + '_split=' + str(split) + '.pkl'

def _temp_experiment_file_name(final_file_name, num_labels):
    directory = helper_functions.remove_suffix(final_file_name, '.pkl')
    return directory + '/num_labels=' + str(num_labels) + '.pkl'

def _load_temp_split_file(final_file_name, num_labels, split):
    split_temp_file = _temp_split_file_name(final_file_name, num_labels, split)
    if not os.path.isfile(split_temp_file):
        return None
    if mpi_utility.is_master():
        print 'found ' + split_temp_file + ' - loading'
    return helper_functions.load_object(split_temp_file)

def _load_temp_experiment_file(final_file_name, num_labels):
    experiment_temp_file = _temp_experiment_file_name(final_file_name, num_labels)
    if not os.path.isfile(experiment_temp_file):
        return None
    if mpi_utility.is_master():
        print 'found ' + experiment_temp_file + ' - loading'
    return helper_functions.load_object(experiment_temp_file)

def _delete_temp_split_files(file, num_labels, num_splits):
    #for i in range(num_splits):
        #helper_functions.delete_file(self._temp_split_file_name(file, i))
    helper_functions.delete_file(_temp_split_file_name(file, num_labels, num_splits))

def _delete_temp_experiment_files(file, num_labels):
    for i in num_labels:
        helper_functions.delete_file(_temp_experiment_file_name(file, i))

def _delete_temp_folder(file):
    folder = helper_functions.remove_suffix(file, '.pkl')
    helper_functions.delete_dir_if_empty(folder)

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
        self.method_experiment_manager_class = getattr(configs,
                                                       'method_experiment_manager_class',
                                                       MethodExperimentManager)

    def run_experiments(self):
        for configs in self.configs.config_list:
            exp_manager = self.method_experiment_manager_class(configs)
            exp_manager.run_experiments()

class MethodExperimentManager(ExperimentManager):
    def __init__(self,configs=None):
        super(MethodExperimentManager,self).__init__(configs)
        pass





    def run_experiments(self):
        data_file = self.configs.data_file
        data_and_splits = helper_functions.load_object(data_file)
        data_and_splits.data.repair_data()
        assert self.configs.num_splits <= len(data_and_splits.splits)
        data_and_splits.labels_to_keep = self.configs.labels_to_keep
        data_and_splits.labels_to_not_sample = self.configs.labels_to_not_sample
        data_and_splits.target_labels = self.configs.target_labels
        data_and_splits.data.repair_data()
        results_file = self.configs.results_file
        comm = mpi_utility.get_comm()
        if os.path.isfile(results_file):
            if mpi_utility.is_group_master():
                print results_file + ' already exists - skipping'
            return            
        if mpi_utility.is_group_master():
            hostname = helper_functions.get_hostname()
            print '(' + hostname  + ') Running experiments: ' + results_file
        learner = self.configs.learner
        learner.run_pre_experiment_setup(data_and_splits)
        num_labels = len(self.configs.num_labels)
        num_splits = self.configs.num_splits
        #method_results = results.MethodResults(n_exp=num_labels, n_splits=num_splits)
        method_results = self.configs.method_results_class(n_exp=num_labels, n_splits=num_splits)
        for i, nl in enumerate(self.configs.num_labels):
            method_results.results_list[i].num_labels = nl

        split_idx = self.configs.split_idx
        if split_idx is not None:
            num_labels_list = list(itertools.product(range(num_labels), [split_idx]))
        else:
            num_labels_list = list(itertools.product(range(num_labels), range(num_splits)))

        shared_args = (self, results_file, data_and_splits, method_results)
        args = [shared_args + (i_labels, split) for i_labels,split in num_labels_list]
        if self.configs.use_pool:
            pool = multiprocessing_utility.LoggingPool(processes=self.configs.pool_size)
            all_results = pool.map(_run_experiment, args)
        else:
            all_results = [_run_experiment(a) for a in args]
        for curr_results,s in zip(all_results,num_labels_list):
            if curr_results is None:
                continue
            i_labels, split = s
            method_results.set(curr_results, i_labels, split)

        method_results.configs = self.configs
        if self.configs.should_load_temp_data:
            helper_functions.save_object(results_file,method_results)
            for i_labels, split in num_labels_list:
                num_labels = self.configs.num_labels[i_labels]
                _delete_temp_split_files(results_file, num_labels, split)
            _delete_temp_folder(results_file)

def _run_experiment(args):
    return _run_experiment_args(*args)

def _run_experiment_args(self, results_file, data_and_splits, method_results, i_labels, split):
    num_labels = self.configs.num_labels[i_labels]
    s = str(num_labels) + '-' + str(split)
    curr_results = _load_temp_split_file(results_file, num_labels, split)
    if curr_results:
        return curr_results
    #print 'num_labels-split: ' + s
    temp_file_name = _temp_split_file_name(results_file, num_labels, split)
    temp_dir_root = helper_functions.remove_suffix(temp_file_name, '.pkl')
    temp_dir = temp_dir_root + '/CV-temp/'
    curr_data = data_and_splits.get_split(split, num_labels)
    learner = self.configs.learner
    curr_learner = copy.deepcopy(learner)
    curr_learner.temp_dir = temp_dir
    curr_results = curr_learner.train_and_test(curr_data)
    if mpi_utility.is_group_master():
        helper_functions.save_object(_temp_split_file_name(results_file,num_labels,split),curr_results)
        helper_functions.delete_dir_if_exists(temp_dir_root)
    if mpi_utility.is_group_master():
        if hasattr(curr_learner, 'best_params'):
            print s + '-' + str(curr_learner.best_params) + ' Error: ' + str(curr_results.compute_error(self.configs.loss_function))
        else:
            print s + ' Done'
    return curr_results
