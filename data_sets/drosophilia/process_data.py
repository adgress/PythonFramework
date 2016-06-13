import numpy as np
from data_sets import create_data_set
from utility import array_functions
from utility import helper_functions

x_file_name = 'SpecificStages-truth-feats.csv'
y_file_name = 'SpecificStages-truth.csv'

_, y = create_data_set.load_csv(y_file_name, True, dtype='str', delim='\t')
y = y[1:,:]
id_to_y = dict((yi[0], int(yi[3])) for yi in y)
pass

feature_names, feats = create_data_set.load_csv(x_file_name, True, dtype='str', delim=str('\t'))
feats = feats[1:,:]
ids = feats[:,0]
feats = np.asarray(feats, dtype='float')


x = feats[:,1:]
y = np.zeros((x.shape[0],1))
for idx, i in enumerate(ids):
    if i in id_to_y:
        y[idx] = id_to_y[i]
    else:
        print 'missing id'
        y[idx] = -1

data = (x,y)
helper_functions.save_object('processed_data.pkl', data)

pass
