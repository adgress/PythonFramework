from configs import base_configs as bc
from utility import helper_functions
from utility import array_functions
import matplotlib.pyplot as plt
from scipy.misc import imread

#data_file_dir = 'boston_housing-13(transfer)'
#data_file_dir = 'synthetic_linear_reg500-10-1-nnz=4'
#data_file_dir = 'kc_housing'
#data_file_dir = 'concrete-7'
#data_file_dir = 'wine-red'
#data_file_dir = 'bike_sharing-feat=1'
#data_file_dir = 'pair_data_82_83'
data_file_dir = 'climate-month'
#data_file_dir = 'irs-income'
#data_file_dir = 'pollution-[3 4]-500-norm'

plot_climate = True
dot_sizes = 70
def vis_data():
    s = 'data_sets/' + data_file_dir + '/raw_data.pkl'
    data = helper_functions.load_object(s)
    x = data.x
    y = data.y
    if plot_climate:
        img_path = 'C:/PythonFramework/far_transfer/figures/climate-terrain.png'
        image = imread(img_path)

        titles = ['Max Temperature: January', 'Max Temperature: April']
        label_idx = [0, 4]

    titles = ['','']
    label_idx =  [0, 1]
    for i, title in zip(label_idx, titles):
        plt.close()
        I = data.data_set_ids == i
        fig = plt.figure(4)
        array_functions.plot_heatmap(x[I,:], y[I], sizes=dot_sizes, fig=fig, title=title)
        if plot_climate:
            plt.imshow(image, zorder=0, extent=[-90, -78, 33.5, 38])
            array_functions.move_fig(fig, 1400, 600)
        plt.show(block=True)

    for i in range(data.p):
        xi = x[:, i]
        title = 'Feature Names Missing'
        if data.feature_names is not None:
            title = data.feature_names[i]
        array_functions.plot_2d(xi, y, data_set_ids=data.data_set_ids, title=title)
        pass

    pass

if __name__ == '__main__':
    vis_data()