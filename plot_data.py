from configs import base_configs as bc
from utility import helper_functions
from utility import array_functions
import matplotlib.pyplot as plt
from scipy.misc import imread
from methods import method
from data import data as data_lib
import numpy as np

#data_file_dir = 'boston_housing-13(transfer)'
#data_file_dir = 'synthetic_linear_reg500-10-1-nnz=4'
#data_file_dir = 'kc_housing'
#data_file_dir = 'concrete-7'
#data_file_dir = 'wine-red'
#data_file_dir = 'bike_sharing-feat=1'
#data_file_dir = 'pair_data_82_83'
#data_file_dir = 'irs-income'
#data_file_dir = 'pollution-[3 4]-500-norm'

#data_file_dir = 'zillow-traffic'
data_file_dir = 'climate-month'

plot_climate = False
plot_gradients = True
plot_features = False
dot_sizes = 70
def vis_data():
    s = 'data_sets/' + data_file_dir + '/raw_data.pkl'
    data = helper_functions.load_object(s)
    x = data.x
    y = data.y
    titles = ['', '']
    if plot_climate:
        img_path = 'C:/PythonFramework/far_transfer/figures/climate-terrain.png'
        image = imread(img_path)
        label_idx = [0, 4]
    titles = ['Max Temperature Gradient: January', 'Max Temperature Gradient: April']

    label_idx =  [0, 1]
    if plot_features:

        for i in range(data.p):
            xi = x[:, i]
            title = 'Feature Names Missing'
            if data.feature_names is not None:
                title = data.feature_names[i]
            array_functions.plot_2d(xi, y, data_set_ids=data.data_set_ids, title=title)
    else:
        for i, title in zip(label_idx, titles):
            #plt.close()
            I = data.data_set_ids == i
            if plot_gradients:
                g = estimate_gradients(x, y, I)
                #g = np.log(g)
                #g -= g.min()
                #g += g.max()/10.0
                #g /= g.max()
                if i == 0:
                    g -= g.min()
                    g /= g.max()
                    g = np.sqrt(g)
                else:
                    g -= g.min()
                    g /= g.max()
                    g **= 1
                #array_functions.plot_heatmap(g, sizes=dot_sizes, fig=fig, title=title)
                fig = plt.figure(i)
                plt.title(title)
                plt.axis('off')
                plt.imshow(g)
                array_functions.move_fig(fig, 750, 400)
                #plt.show(block=False)
            else:
                fig = plt.figure(4)
                array_functions.plot_heatmap(x[I, :], y[I], sizes=dot_sizes, fig=fig, title=title)
                if plot_climate:
                    plt.imshow(image, zorder=0, extent=[-90, -78, 33.5, 38])
                    array_functions.move_fig(fig, 1400, 600)
        plt.show(block=True)

    pass

def estimate_gradients(x, y, I):
    range = x.max(0) - x.min(0)
    #x = (x - x.min(0)) / range
    x = x[I, :]
    y = y[I]
    data = data_lib.Data(x, y)
    data.set_train()
    data.is_regression = True
    nw = method.NadarayaWatsonMethod()
    nw.train_and_test(data)
    num_x0 = 40
    num_x1 = int(num_x0*range[1]/range[0])
    v = np.zeros((num_x0, num_x1))

    x0_vals = np.linspace(x[:, 0].min(), x[:, 0].max(), num_x0)
    x1_vals = np.linspace(x[:, 1].min(), x[:, 1].max(), num_x1)
    x1_vals = x1_vals[::-1]
    v = np.zeros((x1_vals.size, x0_vals.size))

    for idx0, x0 in enumerate(x0_vals):
        for idx1, x1 in enumerate(x1_vals):
            xi = np.asarray([x0, x1])
            d = data_lib.Data(xi[np.newaxis, :], np.asarray([np.nan]))
            v[idx1, idx0] = nw.predict(d).y
        print ''
    gradients = np.gradient(v)
    g = gradients[0]**2 + gradients[1]**2
    g = np.sqrt(g)
    return g


if __name__ == '__main__':
    vis_data()