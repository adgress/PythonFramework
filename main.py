
__author__ = 'Aubrey Gress'

import sys
import os
import importlib

#import configs.base_configs as configs_lib
#import base.project_configs as configs_lib
import base.transfer_project_configs as configs_lib
import boto
import math
from experiment import experiment_manager
from utility import helper_functions
from timer import timer
import matplotlib.pyplot as plt

import numpy as np
from data.data import Data
from loss_functions.loss_function import MeanSquaredError
from loss_functions import loss_function


def run_main():
    pc = configs_lib.ProjectConfigs()
    bc = configs_lib.BatchConfigs(pc)
    batch_exp_manager = experiment_manager.BatchExperimentManager(bc)
    batch_exp_manager.run_experiments()
    #exp_manager = ExperimentManager(configs)
    #exp_exec.run_experiments()

def run_visualization():
    #pc = configs_lib.ProjectConfigs()
    #vis_configs = configs_lib.VisualizationConfigs()

    #plt.plot([1,2,3], [1,4,9], 'rs-',  label='line 2')
    #plt.figure()
    #plt.title(vis_configs.title)

    data_sets = configs_lib.data_sets_for_exps

    max_rows = 3
    n = len(data_sets)
    #fig = plt.figure(len(data_sets))
    fig = plt.figure()
    fig.suptitle('Results')
    num_rows = min(n, max_rows)
    num_cols = math.ceil(float(n) / num_rows)
    for i, data_set_id in enumerate(data_sets):
        subplot_idx = i + 1
        plt.subplot(num_rows,num_cols,subplot_idx)
        axis = [0, 1, 0, .2]
        vis_configs = configs_lib.VisualizationConfigs(data_set_id)
        for file, legend_str in vis_configs.results_files:
            if not os.path.isfile(file):
                print file + ' doesn''t exist - skipping'
                continue
            results = helper_functions.load_object(file)
            #plt.plot([1,2,3], [1,2,3], 'go-', label='line 1', linewidth=2)
            processed_results = results.compute_error_processed(vis_configs.loss_function)
            sizes = results.sizes
            s = legend_str
            if s is None:
                s = results.configs.learner.name_string
            plt.errorbar(sizes,
                         processed_results.means,
                         yerr=[processed_results.lows, processed_results.highs],
                         label=s
            )
            highs = np.asarray(processed_results.means) + np.asarray(processed_results.highs)
            lows = np.asarray(processed_results.means) - np.asarray(processed_results.lows)
            axis[3] = max(axis[3], highs.max() +  .2*lows.min())
        plt.title(vis_configs.title)
        new_x_max = np.max(sizes) + .2*np.min(sizes)
        axis[1] = new_x_max
        show_x_label = num_rows == 1 or subplot_idx > (num_rows-1)*num_cols
        show_y_label = num_cols == 1 or subplot_idx % num_cols == 1

        if show_x_label:
            plt.xlabel(vis_configs.x_axis_string)
        if show_y_label:
            plt.ylabel(vis_configs.y_axis_string)
        #axis[1] *= 2
        axis[3] *= 1.5
        plt.axis(axis)
        plt.legend()
    plt.show()
    x = 1

    '''
    new_x_max = np.max(sizes) + .2*np.min(sizes)
    axis[1] = new_x_max
    plt.xlabel(vis_configs.x_axis_string)
    plt.ylabel(vis_configs.y_axis_string)
    plt.axis(axis)
    plt.legend()
    plt.show()
    '''


if __name__ == "__main__":
    print 'Starting experiments...'
    timer.tic()
    if configs_lib.run_experiments:
        run_main()
    timer.toc()
    if helper_functions.is_laptop():
        run_visualization()





