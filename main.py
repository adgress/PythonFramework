
__author__ = 'Aubrey Gress'

import sys
import os
import importlib

#import configs.base_configs as configs_lib
#import base.project_configs as configs_lib

#import hypothesis_transfer.hypothesis_project_configs as configs_library
#import new_project.new_project_configs as configs_library
#import higher_order_transfer.higher_order_transfer_configs as configs_library

#import base.transfer_project_configs as configs_library
#import far_transfer.far_transfer_project_configs as configs_library
#import mixed_feature_guidance.mixed_features_project_configs as configs_library

#import instance_selection.instance_selectin_project_configs as configs_library
#import active_transfer.active_transfer_project_configs as configs_library

import active.active_project_configs as configs_library
#import active_base.active_base_project_configs as configs_library

configs_lib = configs_library
import boto
import math
from experiment import experiment_manager
from utility import helper_functions, array_functions
from timer import timer
import matplotlib.pyplot as plt
import socket

import numpy as np
from data.data import Data
from loss_functions.loss_function import MeanSquaredError
from loss_functions import loss_function
from utility import mpi_utility
import bisect



def run_experiments(configs=None):
    pc = configs_lib.ProjectConfigs()
    bc = configs_lib.BatchConfigs(pc)
    if configs is not None:
        configs_lib.apply_arguments(configs)
        bc.config_list = [configs]
    batch_exp_manager = experiment_manager.BatchExperimentManager(bc)
    batch_exp_manager.run_experiments()

def get_sized_results(file_name):
    file_name_no_suffix = os.path.basename(helper_functions.remove_suffix(file_name, '.pkl'))
    dir_name = os.path.dirname(file_name)
    all_files = os.listdir(dir_name)
    sized_file_name = file_name_no_suffix + '-num_labels='
    files = []
    results = []
    for s in all_files:
        if sized_file_name in s:
            files.append(dir_name + '/' + s)
            results.append(helper_functions.load_object(dir_name + '/' + s))
    return results

def combine_results(results, sized_results):
    for r in sized_results:
        for idx, size in enumerate(r.sizes):
            if size in results.sizes:
                print 'Duplicate size found - using results in original file'
                continue
            results_idx = bisect.bisect_left(results.sizes, size)
            results.results_list.insert(results_idx, r.results_list[idx])
            pass
    return results

def create_table():
    vis_configs = configs_lib.VisualizationConfigs()
    viz_params = configs_lib.viz_params
    n = len(viz_params)

    if getattr(vis_configs, 'figsize', None):
        fig = plt.figure(figsize=vis_configs.figsize)
    else:
        fig = plt.figure()
    # fig.suptitle('Results')
    # num_rows = min(n, configs_lib.max_rows)
    cell_text = [[np.nan]*len(vis_configs.results_files) for i in range(len(viz_params))]
    cols = []
    rows = []
    size_to_vis = vis_configs.size_to_vis
    baseline_perf = []
    all_perf = []
    data_names = []
    for data_set_idx, curr_viz_params in enumerate(viz_params):
        vis_configs = configs_lib.VisualizationConfigs(**curr_viz_params)
        param_text = []
        if len(rows) <= data_set_idx:
            rows.append(vis_configs.results_dir)
        method_idx = 0
        mean_perf = []

        # Used for column names if not provided by users
        data_names.append(vis_configs.title)
        for file, legend_str in vis_configs.results_files:
            if len(cols) <= method_idx:
                cols.append(legend_str)
            if not os.path.isfile(file):
                print file + ' doesn''t exist - skipping'
                #assert False, 'Creating Table doesn''t work with missing files'
                cell_text[data_set_idx][method_idx] = 'Missing'
                mean_val = np.nan
            else:
                results = helper_functions.load_object(file)
                sized_results = get_sized_results(file)
                results = combine_results(results, sized_results)
                processed_results = results.compute_error_processed(vis_configs.loss_function, normalize_output=True)
                sizes = results.sizes
                #assert size_to_vis in sizes
                #size_idx = array_functions.find_first_element(sizes, size_to_vis)
                size_idx = 1
                # sizes = sizes[0:4]
                s = legend_str
                if s is None:
                    s = results.configs.learner.name_string
                highs = np.asarray(processed_results.means) + np.asarray(processed_results.highs)
                lows = np.asarray(processed_results.means) - np.asarray(processed_results.lows)
                mean_val = processed_results.means[size_idx]
                var = (highs-lows)[size_idx]/2
            latex_str = '-'
            if mean_val < 1000:
                #latex_str = '%.1f \\pm %.1f' % (mean_val, var)
                latex_str = '%.3f (%.2f)' % (mean_val, var)
                #latex_str = '%.2f(%.2f)' % (mean_val, var)
            cell_text[data_set_idx][method_idx] = latex_str
            if method_idx == vis_configs.baseline_idx:
                baseline_perf.append(mean_val)
            mean_perf.append(mean_val)
            method_idx += 1
        all_perf.append(np.asarray(mean_perf))
        #cell_text.append(param_text)
    relative_improvement = np.zeros((len(all_perf), all_perf[0].size))


    # Create table
    method_names_for_table = vis_configs.method_names_for_table
    latex_text = ''
    for method_name in method_names_for_table:
        latex_text += ' & ' + method_name
    latex_text += '\\\\ \hline \n'
    #latex_text = ' & Ours: Linear & Target Only & LLGC & Reweighting & Offset & SMS & Stacking & Ours with Stacking\\\\ \hline \n'

    # If data names are provided, use them instead of ones in config file
    if vis_configs.data_names_for_table is not None:
        data_names = vis_configs.data_names_for_table
    for row_idx, row_str in enumerate(cell_text):
        latex_text += data_names[row_idx] + ' & '
        for i, cell_str in enumerate(row_str):
            latex_text += ' $' + str(cell_str) + '$'
            if i != len(row_str) - 1:
                latex_text += ' &'
        latex_text += ' \\\\ \\hline\n'
    print latex_text

    # If we don't want the "baseline improvement" row
    if vis_configs.baseline_idx is not None:
        for i in range(relative_improvement.shape[0]):
            #relative_improvement[i, :] = (baseline_perf[i] - all_perf[i]) / baseline_perf[i]
            relative_improvement[i, :] = (baseline_perf[i] - all_perf[i]) / baseline_perf[i]
        mean_relative_improvement = ''
        for ri in relative_improvement.T:
            v = ri[np.isfinite(ri)].mean() * 100
            mean_relative_improvement += ('$%.2f$ & ' % v)
        print 'relative improvement: ' + mean_relative_improvement

    fig, axs = plt.subplots()
    axs.axis('tight')
    axs.axis('off')
    the_table = axs.table(
        cellText=cell_text,
        rowLabels=rows,
        colLabels=cols,
        loc='center'
    )
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(10)
    plt.show()
    print ''

def run_visualization():
    vis_configs = configs_lib.VisualizationConfigs()
    #data_sets = configs_lib.data_sets_for_exps
    #n = len(data_sets)
    viz_params = configs_lib.viz_params
    n = len(viz_params)

    if getattr(vis_configs, 'figsize', None):
        fig = plt.figure(figsize=vis_configs.figsize)
    else:
        fig = plt.figure()
    #fig.suptitle('Results')
    #num_rows = min(n, configs_lib.max_rows)
    num_rows = min(n, vis_configs.max_rows)
    num_cols = math.ceil(float(n) / num_rows)
    markers = [
        's', '*', '>', '^', 'v', 'X', 'P', 'd', '*'
    ]
    for config_idx, curr_viz_params in enumerate(viz_params):
        subplot_idx = config_idx + 1
        plt.subplot(num_rows,num_cols,subplot_idx)
        axis = [0, 1, np.inf, -np.inf]
        vis_configs = configs_lib.VisualizationConfigs(**curr_viz_params)
        sizes = None
        min_x = np.inf
        max_x = -np.inf
        marker_idx = -1
        is_file_missing = False
        for file, legend_str in vis_configs.results_files:
            if not os.path.isfile(file):
                is_file_missing = True
                print file + ' doesn''t exist - skipping'
                assert len(viz_params) == 1 or \
                       vis_configs.show_legend_on_all or \
                       vis_configs.show_legend_on_missing_files  or \
                       not vis_configs.crash_on_missing_files, \
                    'Just to be safe, crashing because files are missing'
                continue
            marker_idx += 1
            results = helper_functions.load_object(file)
            sized_results = get_sized_results(file)
            sizes_to_plot = vis_configs.sizes_to_use
            if sizes_to_plot is not None:
                sizes_to_plot = set(sizes_to_plot)
            results = combine_results(results, sized_results)
            to_remove = list()
            for j, s in enumerate(results.sizes):
                if sizes_to_plot is not None and s not in sizes_to_plot:
                    to_remove.append(j)
            for j in reversed(to_remove):
                del results.results_list[j]
            #results.results_list = results.results_list[~to_remove]
            if len(results.sizes) == 0:
                print file + ' has no results for sizes ' + str(sizes_to_plot) + ', skipping'

            #plt.plot([1,2,3], [1,2,3], 'go-', label='line 1', linewidth=2)
            processed_results = results.compute_error_processed(
                vis_configs.loss_function,
                vis_configs.results_features,
                vis_configs.instance_subset,
                normalize_output=True
            )
            sizes = results.sizes

            #sizes = sizes[0:4]
            min_x = min(min_x, sizes.min())
            max_x = max(max_x, sizes.max())
            s = legend_str
            if s is None:
                s = results.configs.learner.name_string
            print 'Plotting: ' + file
            print 'Mean Errors: ' + str(processed_results.means)
            plt.errorbar(sizes,
                         processed_results.means,
                         yerr=[processed_results.lows, processed_results.highs],
                         label=s,
                         marker=markers[marker_idx],
                         markersize=8
            )
            highs = np.asarray(processed_results.means) + np.asarray(processed_results.highs)
            lows = np.asarray(processed_results.means) - np.asarray(processed_results.lows)
            axis[3] = max(axis[3], highs.max() +  .2*lows.min())
            axis[2] = min(axis[2], .9*lows.min())
        if sizes is None:
            print 'Empty plot - skipping'
            continue
        plt.title(vis_configs.title, fontsize=vis_configs.fontsize)
        axis_range = max_x - min_x
        axis[1] = max_x + .1*axis_range
        axis[0] = min_x - .1*axis_range
        #show_x_label = num_rows == 1 or subplot_idx > (num_rows-1)*num_cols
        #show_x_label = num_rows == 1 or subplot_idx == 8
        #show_x_label = subplot_idx == 9
        show_x_label = subplot_idx == 8
        show_y_label = num_cols == 1 or subplot_idx % num_cols == 1 or vis_configs.always_show_y_label

        if show_x_label:
            plt.xlabel(vis_configs.x_axis_string)
        if show_y_label:
            plt.ylabel(vis_configs.y_axis_string)
        #axis[1] *= 2
        axis[3] *= 1
        ylims = getattr(vis_configs,'ylims',None)
        if ylims is not None:
            axis[2] = ylims[0]
            axis[3] = ylims[1]
        plt.axis(axis)
        if config_idx == 2 or vis_configs.show_legend_on_all or len(viz_params) == 1\
                or (vis_configs.show_legend_on_missing_files and is_file_missing):
            plt.legend(loc='upper right', fontsize=vis_configs.fontsize)
    #fig.tight_layout(rect=[.05,.05,.95,.95])
    if getattr(vis_configs,'borders',None):
        left,right,top,bottom = vis_configs.borders
        fig.subplots_adjust(left=left,right=right,top=top,bottom=bottom)
    if vis_configs.use_tight_layout:
        plt.tight_layout()
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

def run_main(num_labels=None, split_idx=None, no_viz=None, configs=None, comm=None):
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
        if mpi_utility.is_group_master():
            print '(' + socket.gethostname() + ')''Process ' + str(comm.Get_rank()) + ': Starting experiments...'
    else:
        print 'Starting experiments...'
    if mpi_utility.is_group_master():
        timer.tic()
    if configs_lib.run_experiments:
        run_experiments(configs)
    if mpi_utility.is_group_master():
        timer.toc()
    if helper_functions.is_laptop():
        import winsound
        winsound.Beep(440, 1000)
    if helper_functions.is_laptop() and not arguments.no_viz and MPI.COMM_WORLD.Get_size() == 1:
        vis_configs = configs_lib.VisualizationConfigs()
        if vis_configs.vis_table:
            create_table()
        else:
            run_visualization()

if __name__ == "__main__":
    run_main()






