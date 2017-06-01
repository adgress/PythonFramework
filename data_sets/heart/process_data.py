import numpy as np
import scipy
from data_sets import create_data_set
from data import data as data_lib
from utility import helper_functions


file = 'SAheart.data.txt'
all_field_names, data = create_data_set.load_csv(file, has_field_names=True,dtype='string',delim=str(','))
data[data == 'Present'] = '1'
data[data == 'Absent'] = '0'
data = data[:, 1:]
data = data.astype(np.float)
data = data_lib.Data(data[:, :-1], data[:, -1])
data.set_train()
data.set_target()
helper_functions.save_object('raw_data.pkl', data)
print ''
