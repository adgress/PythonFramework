import far_transfer_project_configs as configs_lib
from experiment.experiment_manager import MethodExperimentManager
import numpy as np
from timer.timer import tic, toc
from utility.latex import ndarray_to_table
all_data_sets = configs_lib.all_data_sets
names_for_table = configs_lib.names_for_table

def set_default_hyperparamters(learner):
    learner.C = 1
    learner.radius = 1
    learner.sigma_nw = .1
    learner.sigma_tr = .1
    learner.cv_params = dict()

def run_nystrom_timing_test():
    num_labels = 20
    num_splits = 30
    nystrom_percs = [0,.5, .2, .1]
    timings = np.zeros((len(all_data_sets), len(nystrom_percs)))
    for data_idx, data_set in enumerate(all_data_sets):
        for perc_idx, perc in enumerate(nystrom_percs):
            pc = configs_lib.ProjectConfigs(data_set)
            pc.nystrom_percentage = perc
            pc.ft_method = configs_lib.FT_METHOD_GRAPH_NW
            main_configs = configs_lib.MainConfigs(pc)
            set_default_hyperparamters(main_configs.learner)
            experiment_mananager = MethodExperimentManager(main_configs)
            data_file = '../' + main_configs.data_file
            data_and_splits = experiment_mananager.load_data_and_splits(data_file)
            for split_idx in range(num_splits):
                curr_data = data_and_splits.get_split(split_idx, num_labels)
                main_configs.learner.train_and_test(curr_data)
                #avg_time += main_configs.learner.predict_time
                timings[data_idx, perc_idx] += main_configs.learner.predict_time
    timings /= num_splits
    timings *= 1000
    np.set_printoptions(4)
    print names_for_table
    print timings
    print ndarray_to_table((timings,), names_for_table)



if __name__ == '__main__':
    run_nystrom_timing_test()