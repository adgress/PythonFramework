from configs import base_configs as bc
from utility import helper_functions
from utility import array_functions
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.misc import imread
from methods import method
from methods import local_transfer_methods
from configs.base_configs import MethodConfigs
from data import data as data_lib
from data.data import Data, LabeledData
import numpy as np

data_file_dir = 'taxi3'

mat_dim = 39

def vec_to_matrix(v, y):
    x = np.zeros((mat_dim+1, mat_dim+1))
    I = (v*mat_dim).astype(np.int)
    x[I[:, 0], I[:, 1]] = y
    return x

def vis_data():
    s = '../data_sets/' + data_file_dir + '/raw_data.pkl'
    data = helper_functions.load_object(s)
    x = data.x
    y = data.y
    c = MethodConfigs()
    x_mat = vec_to_matrix(x, y)
    #self.set_data_set_defaults('taxi3', source_labels=[1], target_labels=[0], is_regression=True)
    c.source_labels= np.asarray([1])
    c.target_labels= np.asarray([0])
    c.use_validation = True
    I_target = (c.target_labels[0] == data.data_set_ids).nonzero()[0]
    I_to_use = np.random.choice(I_target, 40, replace=False)
    data.y[I_target] = np.nan
    data.y[I_to_use] = data.true_y[I_to_use]
    learner = local_transfer_methods.LocalTransferDeltaNew(c)
    v = 1
    learner.cv_params['sigma_target'] = learner.create_cv_params(-v, v)
    learner.cv_params['sigma_b'] = learner.create_cv_params(-v, v)
    learner.cv_params['sigma_alpha'] = learner.create_cv_params(-v, v)
    #learner.transform = None
    output = learner.train_and_test(data).prediction

    fig = plt.figure(0)
    plt.title('TODO')
    plt.axis('off')

    I_target = data.get_transfer_inds(c.target_labels)
    vals_to_plot = [
        np.abs(output.ft - output.true_y)**1,
        np.abs(output.y_s + output.b - output.true_y)**1,
        output.alpha,
        np.abs(output.y - output.true_y)**1,
    ]
    titles = [
        'Target Function \nError',
        'Adapted Source \nFunction Error',
        'Mixture Function \n',
        'Final Prediction \nError',
    ]
    min_error = min([vals_to_plot[i].min() for i in [0, 1, 3]])
    max_error = max([vals_to_plot[i].max() for i in [0, 1, 3]])
    print output.b
    print output.alpha
    for i, vals in enumerate(vals_to_plot):
        ax = plt.subplot(1, len(vals_to_plot), i+1)

        ax.set_title(titles[i], fontsize=15)
        #array_functions.plot_heatmap(data.x[I_target], vals, fig=fig, make_subplot=False, sizes=20)
        #vals -= min_error
        #vals /= max_error
        vals -= vals.min()
        vals /= vals.max()
        vals_reshaped = np.reshape(vals, (40, 40))
        plt.pcolormesh(vals_reshaped, cmap=cm.gray,shading='flat', norm=None)
        ax.set_xlabel('Latitude')
        if i == 0:
            ax.set_ylabel('Longitude')
        else:
            ax.set_ylabel('')
        plt.xticks([], [])
        plt.yticks([], [])
    array_functions.move_fig(fig, 1200, 400)
    #plt.tight_layout(pad=0, h_pad=0, w_pad=0)
    #plt.wsp
    plt.subplots_adjust(left=0.05, right=0.95, top=0.8, bottom=0.1)
    plt.show(block=True)
    print ''




if __name__ == '__main__':
    vis_data()