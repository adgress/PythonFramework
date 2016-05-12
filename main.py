
__author__ = 'Aubrey Gress'

import sys
import os
import importlib

#import configs.base_configs as configs_lib
#import base.project_configs as configs_lib
#import base.transfer_project_configs as configs_lib
import active.active_project_configs as configs_library
configs_lib = configs_library
import boto
import math
from experiment import experiment_manager
from utility import helper_functions
from timer import timer
import matplotlib.pyplot as plt
import socket

import numpy as np
from data.data import Data
from loss_functions.loss_function import MeanSquaredError
from loss_functions import loss_function
from utility import mpi_utility


def run_experiments():
    pc = configs_lib.ProjectConfigs()
    bc = configs_lib.BatchConfigs(pc)
    batch_exp_manager = experiment_manager.BatchExperimentManager(bc)
    batch_exp_manager.run_experiments()
    #exp_manager = ExperimentManager(configs)
    #exp_exec.run_experiments()

def run_visualization():
    vis_configs = configs_lib.VisualizationConfigs()
    data_sets = configs_lib.data_sets_for_exps
    n = len(data_sets)
    #fig = plt.figure(len(data_sets))

    if getattr(vis_configs, 'figsize', None):
        fig = plt.figure(figsize=vis_configs.figsize)
    else:
        fig = plt.figure()
    #fig.suptitle('Results')
    num_rows = min(n, configs_lib.max_rows)
    num_cols = math.ceil(float(n) / num_rows)
    for i, data_set_id in enumerate(data_sets):
        subplot_idx = i + 1
        plt.subplot(num_rows,num_cols,subplot_idx)
        axis = [0, 1, 0, .2]
        vis_configs = configs_lib.VisualizationConfigs(data_set_id)
        sizes = None
        for file, legend_str in vis_configs.results_files:
            if not os.path.isfile(file):
                print file + ' doesn''t exist - skipping'
                assert vis_configs.show_legend_on_all, 'Just to be safe, set show_legend_on_all=True if files are missing'
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
        if sizes is None:
            print 'Empty plot - skipping'
            continue
        plt.title(vis_configs.title)
        axis_range = np.max(sizes) - np.min(sizes)
        axis[1] = np.max(sizes) + .1*axis_range
        axis[0] = np.min(sizes) - .1*axis_range
        #show_x_label = num_rows == 1 or subplot_idx > (num_rows-1)*num_cols
        #show_x_label = num_rows == 1 or subplot_idx == 8
        show_x_label = subplot_idx == 2
        show_y_label = num_cols == 1 or subplot_idx % num_cols == 1

        if show_x_label:
            plt.xlabel(vis_configs.x_axis_string)
        if show_y_label:
            plt.ylabel(vis_configs.y_axis_string)
        #axis[1] *= 2
        axis[3] *= 1.5
        ylims = getattr(vis_configs,'ylims',None)
        if ylims is not None:
            axis[2] = ylims[0]
            axis[3] = ylims[1]
        plt.axis(axis)
        if i == 0 or vis_configs.show_legend_on_all:
            plt.legend(loc='upper right', fontsize=vis_configs.fontsize)
    #fig.tight_layout(rect=[.05,.05,.95,.95])
    if getattr(vis_configs,'borders',None):
        left,right,top,bottom = vis_configs.borders
        fig.subplots_adjust(left=left,right=right,top=top,bottom=bottom)
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

test_mpi = False

def run_main_args(args):
    #mpi_utility.mpi_print(str(args))
    run_main(*args)

def run_main(num_labels=None, split_idx=None, no_viz=None, comm=None):
    import argparse
    import sys
    #print sys.argv
    parser = argparse.ArgumentParser()
    parser.add_argument('-num_labels', type=int)
    parser.add_argument('-split_idx', type=int)
    parser.add_argument('-no_viz', action='store_true')
    arguments = parser.parse_args(sys.argv[1:])
    if num_labels is not None:
        arguments.num_labels = num_labels
    if split_idx is not None:
        arguments.split_idx = split_idx
    if no_viz is not None:
        arguments.no_viz = no_viz

    configs_lib.comm = comm
    if test_mpi:
        from mpi4py import MPI
        print str(MPI.COMM_WORLD.Get_rank()) + '-' + str(arguments.num_labels) + '-' + str(arguments.split_idx)
        return

    configs_lib.arguments = arguments
    import warnings
    #print 'Ignoring Deprecation Warnings'
    warnings.filterwarnings("ignore",category=DeprecationWarning)

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    if MPI.COMM_WORLD.Get_size() > 1:
        if comm.Get_rank() == 0:
            print '(' + socket.gethostname() + ')''Process ' + str(comm.Get_rank()) + ': Starting experiments...'
    else:
        print 'Starting experiments...'
    if mpi_utility.is_master():
        timer.tic()
    if configs_lib.run_experiments:
        run_experiments()
    if mpi_utility.is_master():
        timer.toc()
    if helper_functions.is_laptop() and not arguments.no_viz:
        run_visualization()

if __name__ == "__main__":
    run_main()






