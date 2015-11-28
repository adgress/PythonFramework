__author__ = 'Evan Racah'
"""
This module contains plotting functionality
"""
from os import system


import matplotlib
matplotlib.use('pdf')
from matplotlib import pyplot as plt
from utility.HelperFunctions import make_dir_for_file_name
from plot_annotations import PlotAnnotations
from line_types import line_class_type_dict
#from learn_grid_search.functions_for_spark import group_results_by


class GraphicalPlot(object):


    def __init__(self, plot_configs_object, spark_context):
        self.spark_context = spark_context
        self.plot_configs = plot_configs_object
        plt.clf()
        self.ax = plt.subplot(111)
        self.plot_annotations = PlotAnnotations(self.plot_configs,self.ax)
        self.line_class = line_class_type_dict[self.plot_configs.graph_type]

    def plot(self, results_objs_lists):
        print results_objs_lists
        line_object = self.line_class(self.plot_annotations, self.ax)

        for line_number, results_obj_list in enumerate(results_objs_lists):

            for setting_number, line_setting in enumerate(self.plot_configs.line_settings):

                for filtered_results_obj_sub_list in self.filter_data(line_setting, list(results_obj_list)):
                    line_object.plot(filtered_results_obj_sub_list, line_setting)

        self.plot_annotations.add_in_results_obj(results_objs_lists)
        self.postprocess_plot()


    def postprocess_plot(self):
        self.plot_annotations.format_graph()

        self.save_plot()
        self.show()

    def show(self):
        if self.plot_configs.show_plot:
            plt.show()

    def save_plot(self):
        if self.plot_configs.save_plot:
            make_dir_for_file_name(self.plot_annotations.file_name)
            print self.plot_annotations.file_name
            plt.savefig(self.plot_annotations.file_name)


    def filter_data(self, line_setting, results_obj_list):

        if 'filter_list_by' in line_setting:
            results_obj_sub_list = group_results_by(line_setting['filter_list_by'], self.spark_context.parallelize(results_obj_list)).collect()

        else:

            results_obj_sub_list = [results_obj_list]

        return results_obj_sub_list






