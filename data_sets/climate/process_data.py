import numpy as np
from data_sets import create_data_set
from utility import array_functions
from utility import helper_functions
from utility.array_functions import find_first_element
import datetime
from collections import OrderedDict

daily_file_names = ['844054.csv', '844056.csv', '844150.csv']
monthly_file_names = ['844233.csv']
use_monthly = True
plot_data = False
y_to_use = 3

def to_date(date_str):
    year = int(date_str[:4])
    if use_monthly:
        print 'TODO: Fix this'
        day = int(date_str[5:])
        month = 1
    else:
        month = int(date_str[4:6])
        day = int(date_str[6:])
    d = datetime.date(year, month, day)
    return d

file_names = daily_file_names
if use_monthly:
    file_names = monthly_file_names

feats_to_keep = ['STATION', 'STATION_NAME', 'LATITUDE', 'LONGITUDE', 'DATE', 'TAVG', 'TMAX', 'TMIN', 'PRCP']

if use_monthly:
    feats_to_keep[1] = 'NAME'

for i, file in enumerate(file_names):
    feat_names_curr, data_curr = create_data_set.load_csv(
        file,
        True,
        dtype='str',
        delim=',',
        num_rows=1000000000
    )
    inds_to_use = np.asarray([j for j in range(feat_names_curr.size) if feat_names_curr[j] in feats_to_keep])
    assert inds_to_use.size == len(feats_to_keep)
    data_curr = data_curr[:, inds_to_use]
    feat_names_curr = feat_names_curr[inds_to_use]
    if i == 0:
        feat_names = feat_names_curr
        data = data_curr
        continue

    unique_stations = np.unique(data[:, find_first_element(feat_names, 'STATION')].astype(np.str))
    curr_stations = data_curr[:, find_first_element(feat_names, 'STATION')].astype(np.str)
    to_remove = array_functions.false(data_curr.shape[0])
    for s in np.unique(curr_stations):
        if s not in unique_stations:
            continue
        print 'Found repeated station, removing: ' + s
        to_remove = to_remove | (curr_stations == s)
    data = np.vstack((data, data_curr[~to_remove,:]))
y_names = ['TAVG', 'TMIN', 'TMAX', 'PRCP']
y_inds = []
for name in y_names:
    y_inds.append(array_functions.find_first_element(feat_names, name))
date_strs = data[:, find_first_element(feat_names, 'DATE')]
prev = ''
date_str_to_idx = dict()
date_ids = np.zeros(data.shape[0])
for i, date_str in enumerate(date_strs):
    date_obj = to_date(date_str)
    date_str_to_idx[date_str] = date_obj.toordinal()
    date_ids[i] = date_obj.toordinal()
date_ids = date_ids.astype(np.int)
y = data[:, y_inds].astype(np.float)

a1 = data[:, find_first_element(feat_names, 'STATION')].astype(np.str)
lat = data[:, find_first_element(feat_names, 'LATITUDE')].astype(np.str)
lon = data[:, find_first_element(feat_names, 'LONGITUDE')].astype(np.str)
locs = np.stack((lon,lat), 1)
locs[locs == 'unknown'] = np.nan
locs = locs.astype(np.float)
series_id = a1


min_date_id = date_ids.min()
max_date_id = date_ids.max()
num_days = max_date_id-min_date_id+1
d = OrderedDict()
for s in series_id:
    d[s] = None
unique_series_ids = np.asarray(list(d))
unique_locs = np.zeros((unique_series_ids.size, 2))
for i, u in enumerate(unique_series_ids):
    unique_locs[i] = locs[find_first_element(series_id, u)]
times_series_vals = -1*np.ones((num_days, unique_series_ids.size, y.shape[1]))
for i, name in enumerate(unique_series_ids):
    I = (series_id==name).nonzero()[0]
    dates_idx = date_ids[I] - min_date_id
    times_series_vals[dates_idx, i, :] = y[I, :]

times_series_vals[times_series_vals < 0] = np.nan

plot_2d = True
plot_multiple_stations = True
y_to_plot = 3
if plot_data:
    if plot_2d:
        for i in range(num_days):
            if use_monthly:
                y_val = times_series_vals[[i, i+4], :, y_to_plot]
            else:
                y_val = times_series_vals[[i,60+i],:,y_to_plot]
                y_val1 = times_series_vals[range(i,i+30),:,y_to_plot].mean(0)
                y_val2 = times_series_vals[range(i+120, i + 150), :, y_to_plot].mean(0)
                y_val = np.stack((y_val1, y_val2), 1).T
            array_functions.plot_heatmap(unique_locs,y_val.T,alpha=1,title=None,sizes=None,share_axis=True)
    elif plot_multiple_stations:
        for i in range(0,400, 10):
            is_in_state = np.arange(i,i+10)
            #y_val = times_series_vals[is_in_state, :800, 1].T
            y_val = times_series_vals[:,is_in_state[:], y_to_plot]
            x_val = range(y_val.shape[0])
            #print unique_series_ids[to_use]
            for i, s in enumerate(unique_series_ids[is_in_state]):
                print str(i) + ': ' + s
            array_functions.plot_2d_sub_multiple_y(np.asarray(x_val), y_val, title=None, sizes=10)
    else:
        for i in range(times_series_vals.shape[1]):
            y_val = times_series_vals[:, i, :]
            x_val = np.arange(y_val.shape[0])
            if not np.isfinite(y_val).sum(0).all():
                print 'skipping - missing labels'
                continue
            print unique_series_ids[i]
            array_functions.plot_2d_sub_multiple_y(np.asarray(x_val), y_val, title=unique_series_ids[i], sizes=10)


data = (unique_locs, times_series_vals[:,:,y_to_use],unique_series_ids)
suffix = y_names[y_to_use]
if use_monthly:
    suffix += '-month'
s = '../climate'
if use_monthly:
    s += '-month'
s += '/processed_data.pkl'
helper_functions.save_object(s, data)

pass
