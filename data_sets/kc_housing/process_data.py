import numpy as np
from data_sets import create_data_set
from utility import array_functions
from utility import helper_functions

file_name = 'kc_house_data.csv'

feat_names, data = create_data_set.load_csv(file_name, True, dtype='str', delim=',')
feats_to_clear = ['id', 'date', 'yr_renovated', 'zipcode', 'lat', 'long']
y_name = 'price'
y_ind =

x = np.asarray(x, dtype='float')
#x = feats[:,1:]
#y = np.zeros((x.shape[0],1))
for idx, i in enumerate(ids):
    if i in id_to_y:
        y[idx] = id_to_y[i]
    else:
        print 'missing id'
        y[idx] = -1

data = (x,y)
helper_functions.save_object('processed_data.pkl', data)

pass
