from configs import base_configs as bc
from utility import helper_functions
from utility import array_functions
import matplotlib.pyplot as plt
from scipy.misc import imread
from methods import method
from methods import local_transfer_methods
from configs.base_configs import MethodConfigs
from data import data as data_lib
from data.data import Data, LabeledData
import numpy as np

data_file_dir = 'taxi3'

def vis_data():
    s = '../data_sets/' + data_file_dir + '/raw_data.pkl'
    data = helper_functions.load_object(s)
    x = data.x
    y = data.y
    c = MethodConfigs()
    #self.set_data_set_defaults('taxi3', source_labels=[1], target_labels=[0], is_regression=True)
    c.source_labels= np.asarray([1])
    c.target_labels= np.asarray([0])
    c.use_validation = True
    I_target = (c.target_labels[0] == data.data_set_ids).nonzero()[0]
    I_to_use = np.random.choice(I_target, 80, replace=False)
    data.y[I_target] = np.nan
    data.y[I_to_use] = data.true_y[I_to_use]
    learner = local_transfer_methods.LocalTransferDeltaNew(c)
    learner.cv_params['sigma_target'] = learner.create_cv_params(-5, 5)
    learner.cv_params['sigma_b'] = learner.create_cv_params(-5, 5)
    learner.cv_params['sigma_alpha'] = learner.create_cv_params(-5, 5)
    #learner.transform = None
    output = learner.train_and_test(data).prediction

    fig = plt.figure(0)
    plt.title('TODO')
    plt.axis('off')

    vals_to_plot = [
        output.y,
        np.abs(output.y - output.true_y),
        np.abs(output.ft - output.true_y),
        np.abs(output.y_s + output.b - output.true_y),
        output.alpha,
        output.b,
    ]
    titles = [
        'Prediction', 'Prediction error', 'f_t error', 'b + y_s error', 'alpha', 'b'
    ]
    for i, vals in enumerate(vals_to_plot):
        ax = plt.subplot(1, len(vals_to_plot), i+1)
        ax.set_title(titles[i])
        array_functions.plot_heatmap(data.x[I_target], vals[I_target], fig=fig, make_subplot=False, sizes=50)

    array_functions.move_fig(fig, 1800, 400)
    plt.show(block=True)
    print ''




if __name__ == '__main__':
    vis_data()