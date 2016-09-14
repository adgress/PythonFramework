import numpy as np
from data_sets import create_data_set
from utility import array_functions
from utility import helper_functions

file_name = 'kc_house_data.csv'

feat_names, data = create_data_set.load_csv(file_name, True, dtype='str', delim=',')
feats_to_clear = ['id', 'date', 'yr_renovated', 'zipcode', 'lat', 'long']
y_name = 'price'
y_ind = array_functions.find_first_element(feat_names, y_name)
y = data[:, y_ind].astype(np.float)
clear_idx = array_functions.find_set(feat_names, feats_to_clear + [y_name])
x = data[:, ~clear_idx]
x = array_functions.remove_quotes(x)
x = x.astype(np.float)

data = (x,y)
helper_functions.save_object('processed_data.pkl', data)

pass
