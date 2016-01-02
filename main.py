
__author__ = 'Aubrey Gress'

import sys
import os
import importlib

#import configs.base_configs as configs_lib
#import base.project_configs as configs_lib
import base.transfer_project_configs as configs_lib

from experiment import experiment_manager
from utility import helper_functions

import matplotlib.pyplot as plt

import numpy as np
from data.data import Data
from loss_functions.loss_function import MeanSquaredError
from loss_functions import loss_function

def run_main():
    pc = configs_lib.ProjectConfigs()
    bc = configs_lib.BatchConfigs()
    batch_exp_manager = experiment_manager.BatchExperimentManager(bc)
    batch_exp_manager.run_experiments()
    #exp_manager = ExperimentManager(configs)
    #exp_exec.run_experiments()

def run_visualization():
    pc = configs_lib.ProjectConfigs()
    vis_configs = configs_lib.VisualizationConfigs()

    #plt.plot([1,2,3], [1,4,9], 'rs-',  label='line 2')
    plt.figure()
    plt.title = vis_configs.title
    axis = [0, 1, 0, 1]
    for i, file in enumerate(vis_configs.results_files):
        if not os.path.isfile(file):
            print file + ' doesn''t exist - skipping'
            continue
        results = helper_functions.load_object(file)
        #plt.plot([1,2,3], [1,2,3], 'go-', label='line 1', linewidth=2)
        processed_results = results.compute_error_processed(vis_configs.loss_function)
        sizes = results.sizes
        plt.errorbar(sizes,
                     processed_results.means,
                     yerr=[processed_results.lows, processed_results.highs],
                     label=results.configs.learner.name_string
        )
        highs = np.asarray(processed_results.means) + np.asarray(processed_results.highs)
        lows = np.asarray(processed_results.means) - np.asarray(processed_results.lows)
        axis[3] = max(axis[3], highs.max() +  .2*lows.min())

    new_x_max = np.max(sizes) + .2*np.min(sizes)
    axis[1] = new_x_max
    plt.xlabel(vis_configs.x_axis_string)
    plt.ylabel(vis_configs.y_axis_string)
    plt.axis(axis)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    run_main()
    if helper_functions.is_laptop():
        run_visualization()





