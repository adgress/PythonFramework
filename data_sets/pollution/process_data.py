import numpy as np
from data_sets import create_data_set
from utility import array_functions
from utility import helper_functions
from utility.array_functions import find_first_element
import datetime
file_name = 'pollution_us_2000_2016.csv'

def to_date(date_str):
    a = date_str.split('-')
    year, month, day = [int(s) for s in a]
    d = datetime.date(year, month, day)
    return d

feat_names, data = create_data_set.load_csv(
    file_name,
    True,
    dtype='str',
    delim=',',
    #num_rows=40000
    num_rows=100000000000
)
y_names = [s + ' Mean' for s in ['NO2', 'O3', 'SO2', 'CO', ]]
y_inds = []
for name in y_names:
    y_inds.append(array_functions.find_first_element(feat_names, name))
to_keep = array_functions.false(data.shape[0])
date_strs = data[:, find_first_element(feat_names, 'Date Local')]
prev = ''
date_str_to_idx = dict()
date_ids = np.zeros(data.shape[0])
for i, date_str in enumerate(date_strs):
    date_obj = to_date(date_str)
    date_str_to_idx[date_str] = date_obj.toordinal()
    date_ids[i] = date_obj.toordinal()
    if prev != date_str:
        to_keep[i] = True
        prev = date_str
data = data[to_keep, :]
date_strs = date_strs[to_keep]
date_ids = date_ids.astype(np.int)
date_ids = date_ids[to_keep]
y = data[:, y_inds].astype(np.float)

#y_sub = y[I, :]

#series_id = data[:, find_first_element(feat_names, 'Site Num')].astype(np.int)
series_id = data[:, find_first_element(feat_names, 'State')] + '-' + \
            data[:, find_first_element(feat_names, 'Site Num')] + '-' + \
            data[:, find_first_element(feat_names, 'City')] + '-' + \
            data[:, find_first_element(feat_names, 'County')]

states = data[:, find_first_element(feat_names, 'State')]
unique_states = np.unique(states)

min_date_id = date_ids.min()
max_date_id = date_ids.max()
num_days = max_date_id-min_date_id+1
unique_series_ids = np.unique(series_id)
#times_series_vals = [np.zeros((unique_series_ids.size, num_days)) for i in range(y.shape[1])]
times_series_vals = -1*np.ones((num_days, unique_series_ids.size, y.shape[1]))
for i, name in enumerate(unique_series_ids):
    I = (series_id==name).nonzero()[0]
    dates_idx = date_ids[I] - min_date_id
    unique_dates = np.unique(dates_idx)
    if dates_idx.size != unique_dates.size:
        print 'repeats: ' + str(dates_idx.size) + ', ' + str(unique_dates.size)
    times_series_vals[dates_idx, i, :] = y[I, :]
    '''
    for j in I:
        print date_strs[j]
    '''
    print 'num_items: ' + str(I.size)
    print 'start: ' + date_strs[I[0]]
    print 'end: ' + date_strs[I[-1]]

times_series_vals[times_series_vals < 0] = np.nan
'''
#for state in unique_states:
for state in unique_series_ids:
    is_in_state = np.asarray([s.find(state) == 0 for s in unique_series_ids])
    is_in_state = is_in_state.nonzero()[0]
    if is_in_state.size > 8:
        is_in_state = is_in_state[:8]
    #y_val = times_series_vals[is_in_state, :800, 1].T
    y_val = times_series_vals[is_in_state[0], :2000, :4]
    x_val = range(y_val.shape[0])
    #print unique_series_ids[to_use]
    for i, s in enumerate(unique_series_ids[is_in_state]):
        print str(i) + ': ' + s
    array_functions.plot_2d_sub_multiple_y(np.asarray(x_val), y_val, title=None, sizes=10)
'''

data = (times_series_vals,unique_series_ids)
helper_functions.save_object('processed_data.pkl', data)

pass
