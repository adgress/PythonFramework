__author__ = 'racah'

from utility import string_functions, HelperFunctions
import abc
import os
import glob

class ResultsSaver(object):
    __metaclass__ = abc.ABCMeta
    def __init__(self, configs_dict, folder_name):
        self.configs_dict = configs_dict
        self.folder_name = folder_name


    def _generate_save_results_filename(self):
        file_string = self.get_display_name_for_params(self.configs_dict['name_params'])
        dir_string = self.get_display_name_for_params(self.configs_dict['dir_params'])
        if dir_string:
            return dir_string + '/' + file_string + '.dat'
        else:
            return file_string + '.dat'

    def get_display_name_for_params(self, params):
        param_dict = {k: self.configs_dict[k] for k in params}
        return string_functions.stringify_key_val_pairs(param_dict)


    @property
    def total_results_save_file_path(self):
        return self.results_save_file_path + self.folder_name + '/' + HelperFunctions.get_date_string() + '/'

    @property
    def results_save_file_name(self):
        return self._generate_save_results_filename()

    @property
    def results_save_file_path(self):
       return self.configs_dict['results_directory_path']

    def save_results(self, results):
            HelperFunctions.save_object(results, self.results_save_string )

    @property
    def results_save_string(self):

        return self.total_results_save_file_path + self.results_save_file_name

    def load_results(self):
        all_or_one = self.configs_dict['load_mode']
        results_dir = self.total_results_save_file_path
        if all_or_one == 'all':
            return [res_obj for res_obj_name in os.listdir(results_dir) if not res_obj_name.startswith('.') for res_obj in HelperFunctions.load_object(results_dir + res_obj_name)   ]
        elif all_or_one == 'newest':
            newest_res_obj_name = max(glob.iglob(results_dir + '*'), key=os.path.getctime)
            print newest_res_obj_name
            return HelperFunctions.load_object(newest_res_obj_name)
        else:
            assert False, "Must specify result upload string, load_mode to either all or newest not its current setting %s"% all_or_one
















