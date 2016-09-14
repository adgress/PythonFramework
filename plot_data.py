from configs import base_configs as bc
from utility import helper_functions
from utility import array_functions

#data_file_dir = 'boston_housing'
#data_file_dir = 'synthetic_linear_reg500-10-1-nnz=4'
#data_file_dir = 'kc_housing'
#data_file_dir = 'concrete'
data_file_dir = 'wine-red'
def vis_data():
    s = 'data_sets/' + data_file_dir + '/raw_data.pkl'
    data = helper_functions.load_object(s)
    x = data.x
    y = data.y
    for i in range(data.p):
        xi = x[:, i]
        array_functions.plot_2d(xi, y)
        pass

    pass

if __name__ == '__main__':
    vis_data()