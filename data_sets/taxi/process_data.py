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


num_bins = 50
just_pickup = True
just_center_data = False
viz = True
save_data = False
use_alternate = True
dot_size = 60
num_files_to_load = np.inf
use_log = True
min_value = -np.inf
max_value = np.inf

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


def get_leading_values(I):
    inds = I.nonzero()[0]
    is_leading = inds[1:] != (inds[:-1] + 1)
    is_leading = np.insert(is_leading, 0, True)
    leading_inds = inds[is_leading]
    return leading_inds

def get_pickup_inds(x, y, time, has_passenger):
    pickup_inds = get_leading_values(has_passenger == 1)
    return pickup_inds


def vec_to_matrix(locs, values, num_bins):
    vals = np.zeros((num_bins, num_bins))
    for i in range(locs.shape[0]):
        vals[locs[i,1], locs[i,0]] = values[i]
    return vals

def to_coordinate(x, x_min, x_max, num_bins):
    bin_width = (x_max - x_min) / num_bins
    bin_center = x_min + bin_width * x + bin_width/2
    return bin_center

def bin_to_coordinates(v, x_bounds, y_bounds, num_bins):
    x_coords = to_coordinate(v[:, 0], x_bounds[0], x_bounds[1], num_bins)
    y_coords = to_coordinate(v[:, 1], y_bounds[0], y_bounds[1], num_bins)
    return np.stack((x_coords, y_coords), 1)


def load_taxi_data(num_files_to_load=np.inf, num_bins=50, use_alternate=True, return_coords=False):
    all_files = [f for f in os.listdir(data_dir) if path.isfile(path.join(data_dir, f))]
    x = []
    y = []
    time = []
    has_passenger = []
    #combined_data_file = 'combined_data.pkl'
    combined_data_file = 'C:/PythonFramework/data_sets/taxi/combined_data.pkl'
    if path.exists(combined_data_file):
        print 'loading combined data...'
        all_data = helper_functions.load_object(combined_data_file)
        print 'done loading data'
    else:
        for i, file in enumerate(all_files):
            if i == num_files_to_load:
                break
            if i >= 535:
                break
            file_data = load_csv(path.join(data_dir, file), has_field_names=False, delim=str(' '))[1]
            y.append(file_data[:,0])
            x.append(file_data[:,1])
            has_passenger.append(file_data[:, 2])
            time.append(file_data[:, 3])
            print i
        all_data = {
            'x': x,
            'y': y,
            'has_passenger': has_passenger,
            'time': time
        }
        print 'saving combined data...'
        helper_functions.save_object(combined_data_file, all_data)
    x = all_data['x']
    y = all_data['y']
    has_passenger = all_data['has_passenger']
    time = all_data['time']
    x_all = np.concatenate(x)
    y_all = np.concatenate(y)
    time_all = np.concatenate(time)

    has_passenger_all = np.concatenate(has_passenger)

    pickup_inds = get_pickup_inds(x_all, y_all, time_all, has_passenger_all)
    if just_pickup:
        x_all = x_all[pickup_inds]
        y_all = y_all[pickup_inds]
        has_passenger_all = has_passenger_all[pickup_inds]
        time_all = time_all[pickup_inds]
    #x_bounds = [-122.45677419354838, -122.38322580645161]
    #y_bounds = [37.738054968287521, 37.816543340380548]

    x_bounds = [-122.48, -122.35]
    y_bounds = [37.7, 37.84]

    #x_bounds = [-np.inf, np.inf]
    #y_bounds = x_bounds
    is_in_range = in_range(x_all, *x_bounds) & in_range(y_all, *y_bounds)
    x_all = x_all[is_in_range]
    y_all = y_all[is_in_range]
    x_all = quantize_loc(x_all, num_bins)
    y_all = quantize_loc(y_all, num_bins)
    time_all = time_all[is_in_range]

    hours = 9*np.ones(time_all.shape)

    get_hour_vec = np.vectorize(get_hour)
    hours = get_hour_vec(time_all)

    '''
    get_day_vec = np.vectorize(get_day)
    days = get_day_vec(time_all)
    '''
    has_passenger_all = has_passenger_all[is_in_range]


    suffix = '3'
    is_morning = (hours == 9)
    is_night = (hours == 18)
    #is_morning = (hours == 6) & (days == 21)
    #is_night = (hours == 18) & (days == 21)
    #is_morning = (days == 21)
    #is_night = (days == 24)
    if use_alternate:
        is_morning = (hours >= 5) & (hours <= 12)
        is_night = (hours >= 17)
        #is_morning = days == 21
        #is_night = days == 24
        #is_morning = (has_passenger_all == 1) & (days == 21) & is_morning
        #is_night = (has_passenger_all == 1) & (days == 21) & is_night
        #is_morning = (has_passenger_all == 1) & (hours == 6)
        #is_night = (has_passenger_all == 1) & (hours == 18)
        suffix = '2'

    suffix += '-' + str(num_bins)
    #print np.unique(days)

    #is_morning = days == 4
    #is_night = days == 8

    day_locs, day_values = count_cars(x_all[is_morning], y_all[is_morning], num_bins)
    night_locs, night_values = count_cars(x_all[is_night], y_all[is_night], num_bins)
    if return_coords:
        day_locs = bin_to_coordinates(day_locs, x_bounds, y_bounds, num_bins)
        night_locs = bin_to_coordinates(night_locs, x_bounds, y_bounds, num_bins)
    '''
    if use_alternate:
        I = (day_values > 0) | (night_values > 0)
        I = I & (day_values > 0) & (night_values > 0)
    else:
        I = (day_values > 5) | (night_values > 5)
        I = I & (day_values > 0) & (night_values > 0)
    relative_diff = np.max(day_values[I] - night_values[I]) / day_values[I]
    '''
    #array_functions.plot_heatmap(day_locs[I], relative_diff, sizes=10, alpha=1, subtract_min=False)
    return day_locs, day_values, night_locs, night_values, suffix


if __name__ == '__main__':
    day_locs, day_values, night_locs, night_values, suffix = load_taxi_data(num_files_to_load, num_bins, use_alternate)

    '''
    all_locs, all_values = count_cars(x_all, y_all, num_bins)
    is_nonzero = all_values > 0
    all_locs = all_locs[is_nonzero,:]
    all_values = all_values[is_nonzero]
    print 'percentage zero: ' + str(1 - float(is_nonzero.sum())/is_nonzero.size)
    '''

    #locations = np.stack((x_all, y_all)).T
    #plot_y = np.ones(locations.shape[0])


    #array_functions.plot_heatmap(all_locs, np.log(all_values), sizes=10, alpha=1, subtract_min=False)
    #day_values[day_values == 0] = 1
    #night_values[night_values == 0] = 1

    for i in range(day_locs.shape[1]):
        day_locs[:, i] = day_locs[:, i] / day_locs[:, i].max()
    for i in range(night_locs.shape[1]):
        night_locs[:, i] = night_locs[:, i] / night_locs[:, i].max()
    if use_log:
        day_values = np.log(day_values)
        night_values = np.log(night_values)
    else:
        suffix += '-noLog'
    if viz:

        fig1 = pl.figure(3)
        I = np.isfinite(day_values)
        I &= array_functions.in_range(day_values, min_value, max_value)
        if just_center_data:
            I = in_range(day_locs[:,0], .2, .8) & in_range(day_locs[:,1], .2, .8)
        array_functions.plot_heatmap(day_locs[I,:], day_values[I], sizes=dot_size, alpha=1, subtract_min=False, fig=fig1)
        pl.title('day values')
        fig2 = pl.figure(4)
        I = np.isfinite(night_values)
        I &= array_functions.in_range(night_values, min_value, max_value)

        if just_center_data:
            I = in_range(night_locs[:, 0], .2, .8) & in_range(night_locs[:, 1], .2, .8)
        array_functions.plot_heatmap(night_locs[I,:], night_values[I], sizes=dot_size, alpha=1, subtract_min=False, fig=fig2)
        pl.title('night values')
        array_functions.move_fig(fig1, 500, 500, 2000, 100)
        array_functions.move_fig(fig2, 500, 500, 2600, 100)
        pl.show(block=True)

    x = np.vstack((day_locs, night_locs))
    '''
    for i in range(x.shape[1]):
        x[:,i] = x[:,i] / x[:,i].max()
    '''
    data_set_ids = np.hstack((np.zeros(day_values.size), np.ones(day_values.size)))

    y = np.hstack((day_values, night_values))

    '''
    if use_alternate:
        I = np.isfinite(y) & (y > 0)
    else:
        I = np.isfinite(y) & (y > 0) & (y > np.log(5))
    '''
    #I[~np.isfinite(y)] = 0
    I = np.isfinite(y)
    I &= array_functions.in_range(y, min_value, max_value)
    if just_center_data:
        I = I & in_range(x[:,0], .2, .8) & in_range(x[:,1], .2, .8)

    data = data_lib.Data(x[I,:], y[I])
    data.data_set_ids = data_set_ids[I]
    print 'n: ' + str(data.n)
    print 'n0: ' + str((data.data_set_ids == 0).sum())
    print 'n1: ' + str((data.data_set_ids == 1).sum())
    if save_data:
        pass
        file_path = '../taxi%s/raw_data.pkl' % suffix
        helper_functions.save_object(file_path, data)
    print ''