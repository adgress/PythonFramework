import numpy as np
from data_sets import create_data_set
from utility import array_functions
from utility import helper_functions
from datetime import date
from matplotlib import pyplot as pl
from data import data as data_lib

try:
    data = helper_functions.load_object('train.pkl')
except:
    file_name = 'train.csv'
    feat_names, data = create_data_set.load_csv(file_name, True, dtype=np.float, delim=',')
    data = data.astype(np.float)
    Y = data[:, 0]
    X = data[:, 1:]
    data = {
        'X': X,
        'Y': Y
    }
    helper_functions.save_object('train.pkl', data)
x = data['X']
x /= 256
y = data['Y']
data = data_lib.Data(x, y)
helper_functions.save_object('raw_data.pkl', data)
pass