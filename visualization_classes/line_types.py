__author__ = 'racah'
import abc
from matplotlib import pyplot as plt
import numpy as np
import collections

class SinglePlot(object):
    __metaclass__ = abc.ABCMeta
    def __init__(self,plot_annotation_obj, axes_object):
        self.ax = axes_object
        self.results_obj_list = None
        self.line_setting = None
        self.line_appearance_args = None
        self.plot_annotation_obj = plot_annotation_obj
        self.colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        self.color_index = 0

    def get_desired_values(self, attrib_name):
        values = []
        assert isinstance(self.results_obj_list, collections.Iterable), 'Results must be in form of an iterbale of iterables of results objects'

        for results_obj in self.results_obj_list:

            if hasattr(results_obj, attrib_name):
                values.append(getattr(results_obj,attrib_name))

        return values

    def get_desired_y_values(self):
        return self.get_desired_values(self.line_setting['y_attribute'])

    def get_desired_x_values(self):
        return self.get_desired_values(self.line_setting['x_attribute'])


    def plot(self, results_obj_list, line_setting):
        self.results_obj_list = results_obj_list
        self.line_setting = line_setting
        self.line_appearance_args = line_setting['line_appearance']
        if not (self._plot() == False):
            self.plot_annotation_obj.add_to_legend(self.legend_name)


    @abc.abstractmethod
    def _plot(self):
        return

    def new_color(self):
        color = self.colors[self.color_index % len(self.colors)]
        self.color_index+=1
        return color



    @property
    def legend_name(self):
        res_obj = list(self.results_obj_list)[0]
        legend_name = ''
        legend_attr_list = self.line_setting['legend_attribute']
        if not isinstance(legend_attr_list, list):
            legend_attr_list = [legend_attr_list]

        for legend_attr_name in legend_attr_list:
            legend_name += str(getattr(res_obj, legend_attr_name))

        return legend_name + self.line_setting['legend_estimator_suffix']







class BoxPlot(SinglePlot):
    def __init__(self, plot_annotation_obj, axes_object):
        super(BoxPlot, self).__init__(plot_annotation_obj, axes_object)

    def get_desired_x_values(self):
        pass

    def _plot(self):
        data = self.get_desired_y_values()
        self.ax.boxplot(data, **self.line_appearance_args)



class BarPlot(SinglePlot):
    def __init__(self, plot_annotation_obj, axes_object):
        super(BarPlot, self).__init__(plot_annotation_obj, axes_object)
        self.number = 0

    def get_desired_y_values(self):
        y_values = self.get_desired_values(self.line_setting['y_attribute'])
        print self.line_setting['y_attribute']
        print y_values
        y_mean = np.mean(y_values)
        y_var = np.var(y_values)
        return y_mean, y_var

    def get_desired_x_values(self):
        pass

    def _plot(self):
        y_mean, y_var = self.get_desired_y_values()
        width = 0.1
        self.ax.bar(width*self.number, y_mean, width, color=self.new_color(), yerr=np.sqrt(y_var), **self.line_appearance_args)
        self.number += 1



class ScatterPlot(SinglePlot):
    def __init__(self,results_obj_list, axes_object):
        super(ScatterPlot,self).__init__(results_obj_list, axes_object)

    def _plot(self):
        x = self.get_desired_x_values()
        y = self.get_desired_y_values()
        if len(x) != len(y):
            return False
        plt.scatter(x,y,color=self.new_color(), **self.line_appearance_args)
        return True




class LinePlot(SinglePlot):
    def __init__(self, plot_annotation_obj, axes_object):
        super(LinePlot,self).__init__(plot_annotation_obj, axes_object)


    def _plot(self):
        x = self.get_desired_x_values()
        y = self.get_desired_y_values()
        self.ax.plot(x,y,color=self.new_color(), **self.line_appearance_args)


class Scatter3DPlot(SinglePlot):
    def __init__(self,plot_annotation_obj, axes_object):
        super(Scatter3DPlot,self).__init__(plot_annotation_obj, axes_object)

    def _plot(self):
        x = self.get_desired_x_values()
        y = self.get_desired_y_values()
        z = self.get_desired_values('z_attribute')
        self.ax.plot(x,y,z, color=self.new_color(), **self.line_appearance_args)






# TODO: sort of broken
#Make sure resultsin form of list
class ErrorBarLine(SinglePlot):
    #'error_bar_description' : 'Error bars represent two standard deviations'
    def __init__(self,plot_annotation_obj, axes_object):
        super(ErrorBarLine,self).__init__(plot_annotation_obj, axes_object)



    def get_desired_x_values(self):
        single_x_value_set = set(self.get_desired_values(self.line_setting['x_attribute']))
        assert len(single_x_value_set) == 1, 'X-values for each point must be the same!'
        return single_x_value_set[0]



    def get_desired_y_values(self):
        y_values = self.get_desired_values(self.line_setting['y_attribute'])
        y_means = np.mean(y_values)
        y_vars = np.var(y_values)
        return y_means, y_vars


    def _plot(self):
        x_values = self.get_desired_x_values()
        y_values, y_variances = self.get_desired_y_values()
        self.ax.errorbar(x_values,y_values, yerr=2*np.sqrt(y_variances),color=self.new_color(), **self.line_appearance_args)





line_class_type_dict = {
        'error_bar_plot': ErrorBarLine,
        'line_plot' : LinePlot,
        'bar_plot' : BarPlot,
        # 'histogram',
        'boxplot' : BoxPlot,
        'scatter': ScatterPlot,
        '3d_scatter' : Scatter3DPlot
    }
