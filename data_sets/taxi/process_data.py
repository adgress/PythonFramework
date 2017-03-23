import os
from os import path
from data_sets.create_data_set import load_csv, quantize_loc
data_dir = 'C:/Users/adgress/Desktop/cabspottingdata'
import numpy as np
from utility import array_functions
from utility.array_functions import normalize, in_range
from utility import helper_functions
from datetime import datetime
import matplotlib.pylab as pl
from data import data as data_lib

def get_hour(x):
    d = datetime.fromtimestamp(x)
    return d.hour

def get_day(x):
    d = datetime.fromtimestamp(x)
    return d.day

def count_cars(x, y, num_bins):
    all_values = np.zeros(num_bins ** 2)
    all_locs = np.zeros((num_bins ** 2, 2))
    idx = 0
    for xi in range(0, num_bins):
        I = x == xi
        for yi in range(0, num_bins):
            num_cars = (I & (y == yi)).sum()
            all_locs[idx, :] = np.asarray([xi, yi])
            all_values[idx] = num_cars
            idx += 1
    return all_locs, all_values

def vec_to_matrix(locs, values, num_bins):
    vals = np.zeros((num_bins, num_bins))
    for i in range(locs.shape[0]):
        vals[locs[i,1], locs[i,0]] = values[i]
    return vals

num_bins = 400
all_files = [f for f in os.listdir(data_dir) if path.isfile(path.join(data_dir, f))]
x = []
y = []
time = []
has_passenger = []
for i, file in enumerate(all_files):
    if i == 100:
        break
    file_data = load_csv(path.join(data_dir, file), has_field_names=False, delim=str(' '))[1]
    y.append(file_data[:,0])
    x.append(file_data[:,1])
    has_passenger.append(file_data[:, 2])
    time.append(file_data[:, 3])
    print i
x_all = np.concatenate(x)
y_all = np.concatenate(y)
time_all = np.concatenate(time)

has_passenger_all = np.concatenate(has_passenger)
x_bounds = [-122.45677419354838, -122.38322580645161]
y_bounds = [37.738054968287521, 37.816543340380548]
is_in_range = in_range(x_all, *x_bounds) & in_range(y_all, *y_bounds)
x_all = x_all[is_in_range]
y_all = y_all[is_in_range]
x_all = quantize_loc(x_all, num_bins)
y_all = quantize_loc(y_all, num_bins)

time_all = time_all[is_in_range]
get_hour_vec = np.vectorize(get_hour)
hours = get_hour_vec(time_all)
get_day_vec = np.vectorize(get_day)
days = get_day_vec(time_all)

has_passenger_all = has_passenger_all[is_in_range]


#is_morning = (hours >= 5) & (hours <= 12)
#is_night = (hours >= 17)
#is_morning = hours == 6
#is_night = hours == 18
is_morning = days == 4
is_night = days == 8
day_locs, day_values = count_cars(x_all[is_morning], y_all[is_morning], num_bins)
night_locs, night_values = count_cars(x_all[is_night], y_all[is_night], num_bins)

I = (day_values > 5) | (night_values > 5)
I = I & (day_values > 0) & (night_values > 0)
relative_diff = np.max(day_values[I] - night_values[I]) / day_values[I]

#array_functions.plot_heatmap(day_locs[I], relative_diff, sizes=10, alpha=1, subtract_min=False)

day_mat = vec_to_matrix(day_locs, day_values, num_bins)
night_mat = vec_to_matrix(night_locs, night_values, num_bins)

'''
all_locs, all_values = count_cars(x_all, y_all, num_bins)
is_nonzero = all_values > 0
all_locs = all_locs[is_nonzero,:]
all_values = all_values[is_nonzero]
print 'percentage zero: ' + str(1 - float(is_nonzero.sum())/is_nonzero.size)
'''

locations = np.stack((x_all, y_all)).T
plot_y = np.ones(locations.shape[0])


#array_functions.plot_heatmap(all_locs, np.log(all_values), sizes=10, alpha=1, subtract_min=False)
'''
fig1 = pl.figure(3)
array_functions.plot_heatmap(day_locs, np.log(day_values), sizes=30, alpha=1, subtract_min=False, fig=fig1)
fig2 = pl.figure(4)
array_functions.plot_heatmap(night_locs, np.log(night_values), sizes=30, alpha=1, subtract_min=False, fig=fig2)
pl.show(block=True)
'''
x = np.vstack((day_locs, night_locs))
for i in range(x.shape[1]):
    x[:,i] = x[:,i] / x[:,i].max()

data_set_ids = np.hstack((np.zeros(day_values.size), np.ones(day_values.size)))

y = np.hstack((day_values, night_values))
y = np.log(y)
I = np.isfinite(y) & (y > 0)
data = data_lib.Data(x[I,:], y[I])
data.data_set_ids = data_set_ids[I]
helper_functions.save_object('raw_data.pkl', data)
print ''