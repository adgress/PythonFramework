from configs import base_configs as bc
from active_transfer import active_transfer_project_configs as configs_lib
from utility import helper_functions
from utility import array_functions
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.misc import imread
from methods import method
from methods import active_methods, instance_selection
from configs.base_configs import MethodConfigs
from data import data as data_lib
from data.data import Data, LabeledData
import numpy as np
from scipy.stats import binned_statistic_2d

data_file_dir = 'synthetic_piecewise'

mat_dim = 39


def vis_data():
    pc = configs_lib.ProjectConfigs(bc.DATA_KC_HOUSING)
    #pc = configs_lib.ProjectConfigs(bc.DATA_CLIMATE_MONTH)
    pc.active_method = configs_lib.ACTIVE_CLUSTER_PURITY
    #pc.active_method = configs_lib.ACTIVE_CLUSTER
    #pc.active_method = configs_lib.ACTIVE_RANDOM
    pc.fixed_sigma_x = False
    pc.no_spectral_kernel = False
    pc.no_f_x = False
    pc.active_items_per_iteration = 10
    use_oracle_target = False

    main_configs = configs_lib.MainConfigs(pc)
    data_file = '../' + main_configs.data_file
    data_and_splits = helper_functions.load_object(data_file)
    data = data_and_splits.get_split(0, 0)
    is_target = data.data_set_ids == main_configs.target_labels[0]
    is_source = data.data_set_ids == main_configs.source_labels[0]
    data.reveal_labels(is_source.nonzero()[0])
    data.type = data_lib.TYPE_TARGET*np.ones(data.n)
    data.type[is_source] = data_lib.TYPE_SOURCE
    x = data.x
    y = data.y


    learner = main_configs.learner
    learner.use_oracle_target = use_oracle_target
    if pc.active_method == configs_lib.ACTIVE_CLUSTER_PURITY and False:
        learner.instance_selector.cv_params['sigma_y'] = [1]
    print 'Experiment: ' + learner.prefix
    results = learner.train_and_test(data)
    queried_data = results.results_list[0].queried_idx
    selected_data = data.get_subset(queried_data)

    fig = plt.figure(0, figsize=(12, 5))
    plt.title('TODO')
    plt.axis('off')

    x1 = data.x[:, 0]
    x1_sel = selected_data.x[:, 0]
    if data.p == 1:
        x2 = data.true_y
        x2_sel = selected_data.true_y
    else:
        assert data.p == 2
        x2 = data.x[:, 1]
        x2_sel = selected_data.x[:, 1]

    plt.subplot(1, 3, 1)
    plt.scatter(x1[is_target], x2[is_target], c='b', s=10)
    plt.scatter(x1_sel, x2_sel, c='r', s=20)


    if data.p == 2:
        plt.subplot(1, 3, 2)

        target_data = data.get_subset(is_target)
        target_data.y = target_data.true_y.copy()

        nw_method = method.NadarayaWatsonMethod()
        y_pred = nw_method.train_and_test(target_data).prediction.y
        means, _, _, _ = binned_statistic_2d(target_data.x[:, 0], target_data.x[:, 1], y_pred, bins=30)
        #means = means[:, ::-1]
        #means = means[::-1, :]
        means[~np.isfinite(means)] = -1
        plt.pcolormesh(means, cmap='RdBu')
        plt.colorbar()

        plt.subplot(1, 3, 3)

        source_data = data.get_subset(is_source)
        source_data.y = source_data.true_y.copy()

        nw_method = method.NadarayaWatsonMethod()
        y_pred = nw_method.train_and_test(source_data).prediction.y
        means, _, _, _ = binned_statistic_2d(source_data.x[:, 0], source_data.x[:, 1], y_pred, bins=30)
        # means = means[:, ::-1]
        # means = means[::-1, :]
        means[~np.isfinite(means)] = -1
        plt.pcolormesh(means, cmap='RdBu')
        plt.colorbar()

    plt.show()




if __name__ == '__main__':
    vis_data()