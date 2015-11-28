__author__ = 'racah'

from utility import HelperFunctions
from matplotlib.font_manager import FontProperties
import numpy as np

class PlotAnnotations(object):
        """
        Struct-style class for containing configs for a single plot
        """
        def __init__(self, plot_configs, axes_object):
            self.plot_configs = plot_configs
            self.legend_list = []
            self.results_obj_list = None
            self.x_label = plot_configs.x_label
            self.y_label = plot_configs.y_label
            self.z_label = plot_configs.z_label
            self.ax = axes_object

        @property
        def file_name(self):
            return self.plot_configs.path_to_store_plot + '/' + HelperFunctions.get_date_string() + '/' + self.plot_title.replace(' ','_').replace('.','').lower() + '.pdf'

        def format_graph(self):
            self.format_legend()
            self.set_plot_captions()

        def set_plot_captions(self):
            self.plot_title = self.generate_title()
            self.ax.set_ylabel(self.y_label)
            self.ax.set_xlabel(self.x_label)
            if self.z_label:
                self.ax.set_z_label(self.z_label)
            self.ax.set_title(self.plot_title,fontdict = {'fontsize': self.plot_configs.title_size})

        def add_in_results_obj(self,obj_list):
            self.results_obj_list = obj_list

        def generate_title(self):
            plot_title = self.plot_configs.plot_title

            plot_title = plot_title.split('$')
            for i in [x for x in range(len(plot_title)) if x%2 != 0]:
                plot_title[i] = str(self.get_value(self.results_obj_list, plot_title[i]))


            return ''.join(plot_title)

        def get_value(self, res_obj_list, attr):
            tot = []
            for list in res_obj_list:
                for res_obj in list:
                    if hasattr(res_obj, attr):
                        obj_attr = getattr(res_obj, attr)
                        if isinstance(obj_attr, str):
                            return obj_attr
                        else:
                            tot.append(obj_attr)

            if len(tot) > 0:
                return int(np.mean(tot))
            else:
                return 'no'



        def add_to_legend(self,name):
            self.legend_list.append(name)

        def format_legend(self):

            ax = self.ax

            # Shrink current axis's height by 10% on the bottom
            box = ax.get_position()
            ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])

            # Put a legend below current axis
            ax.legend(self.legend_list, loc='upper center', bbox_to_anchor=(0.5, -0.1),
                        fancybox=True, shadow=True, ncol=5, prop = {'size': 8})
