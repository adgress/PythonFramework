import numpy as np
from data_sets import create_data_set
from utility import array_functions
from utility import helper_functions
from utility.array_functions import find_first_element
import datetime
import math

file_name = 'trip.csv'
station_file_name = 'station.csv'

def to_date(date_str):
    a = date_str.split(' ')[0]
    a = a.split('/')
    month, day, year = [int(s) for s in a]
    d = datetime.date(year, month, day)
    return d

def create_stations(file):
    feat_names, data = create_data_set.load_csv(
        file,
        True,
        dtype='str',
        delim=',',
        num_rows=1000000000
    )
    names = data[:, array_functions.find_first_element(feat_names, 'station_id')]
    locs = data[:, array_functions.find_set(feat_names, ['long', 'lat'])]
    return names, locs.astype(np.float)

station_names, station_locs = create_stations(station_file_name)

feat_names, data = create_data_set.load_csv(
    file_name,
    True,
    dtype='str',
    delim=',',
    num_rows=1000000000
)
y_names = ['tripduration']
y_inds = array_functions.find_set(feat_names, y_names).nonzero()[0]
date_strs = data[:, find_first_element(feat_names, 'starttime')]
date_str_to_idx = dict()
date_ids = np.zeros(data.shape[0])
for i, date_str in enumerate(date_strs):
    date_obj = to_date(date_str)
    date_str_to_idx[date_str] = date_obj.toordinal()
    date_ids[i] = date_obj.toordinal()
date_ids = date_ids.astype(np.int)
y = data[:, y_inds].astype(np.float)

#y_sub = y[I, :]

#series_id = data[:, find_first_element(feat_names, 'Site Num')].astype(np.int)
a1 = data[:, find_first_element(feat_names, 'from_station_id')].astype(np.str)
a2 = data[:, find_first_element(feat_names, 'to_station_id')].astype(np.str)
#series_id = np.asarray([a + '-' + b for a,b in zip(a1,a2)])
series_id = a1

min_date_id = date_ids.min()
max_date_id = date_ids.max()
num_days = max_date_id-min_date_id+1
unique_series_ids = np.unique(series_id)
#times_series_vals = [np.zeros((unique_series_ids.size, num_days)) for i in range(y.shape[1])]
times_series_vals = 0*np.ones((num_days, unique_series_ids.size))
#times_series_vals = -1*np.ones((num_days, unique_series_ids.size))
for i, name in enumerate(unique_series_ids):
    I = (series_id==name).nonzero()[0]
    dates_idx = date_ids[I] - min_date_id
    unique_dates, counts = np.unique(dates_idx, return_counts=True)
    if dates_idx.size != unique_dates.size:
        print 'repeats: ' + str(dates_idx.size) + ', ' + str(unique_dates.size)
    #times_series_vals[dates_idx, i, :] = y[I, :]    ]
    for d in unique_dates:
        is_date = dates_idx == d
        times_series_vals[d, i] = math.log(y[I[is_date]].sum())
    #times_series_vals[unique_dates, i] = counts
    '''
    for d in unique_dates:
        times_series_vals[d,i] = y[I[dates_idx == d]].mean()
    pass
    '''
    '''
    for j in I:
        print date_strs[j]
    '''
    '''
    print 'num_items: ' + str(I.size)
    print 'start: ' + date_strs[I[0]]
    print 'end: ' + date_strs[I[-1]]
    '''

has_loc = array_functions.false(unique_series_ids.size)
for i, id in enumerate(unique_series_ids):
    has_loc[i] = id in station_names
times_series_vals = times_series_vals[:, has_loc]
unique_series_ids = unique_series_ids[has_loc]
date_idx = 0
for i in range(0,num_days, 5):
    x = station_locs
    y = times_series_vals[i:120:28,:]
    array_functions.plot_heatmap(x, y.T, title=None, sizes=30)


data = (times_series_vals,unique_series_ids)
helper_functions.save_object('processed_data.pkl', data)

pass
