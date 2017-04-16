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
from data_sets import create_data_set
from methods import transfer_methods, far_transfer_methods
from configs import base_configs
from data_sets.create_data_split import DataSplitter
from copy import deepcopy
from loss_functions import loss_function

def replace_invalid_strings(x):
    try:
        float(x)
    except:
        return 'nan'
    return x

def remove_quotations(x):
    return x[1:-1]

vec_remove_quotations = np.vectorize(remove_quotations)
vec_replace = np.vectorize(replace_invalid_strings)
use_zipcode_data = False
apply_log = use_zipcode_data
viz = False
run_state_tests = False
combine_with_traffic_data = True

def get_zipcode_locations():
    file = '../zipcodes/zipcodes.txt'
    fields, zipcode_data = create_data_set.load_csv(file, has_field_names=True, dtype=np.float)
    locs = zipcode_data[:, [2,1]]
    zip_codes = zipcode_data[:,0].astype(np.int)
    zipcode_location_map = dict()
    for z, loc in zip(zip_codes, locs):
        zipcode_location_map[z] = loc
    return zipcode_location_map

def create_transfer_data(locations, pricing_data, I, apply_log=False):
    x_all = np.vstack((locations[I, :], locations[I, :]))
    y_all = np.concatenate((pricing_data[I, 0], pricing_data[I, 1]))
    if apply_log:
        y_all = np.log(y_all)
    else:
        print 'not taking log of labels!'
    #y_all /= y_all[np.isfinite(y_all)].max()
    data_set_ids = np.concatenate((np.zeros(I.sum()), np.ones(I.sum())))
    data = data_lib.Data(x_all, y_all)
    data.data_set_ids = data_set_ids
    data.is_regression = True
    return data

def combine_data(x1, y1, x2, y2):
    x = np.vstack((x1, x2))
    y = np.concatenate((y1, y2))
    data_set_ids = np.concatenate((np.zeros(y1.size), np.ones(y2.size)))
    data = data_lib.Data(x, y)
    data.data_set_ids = data_set_ids
    data.is_regression
    return data


if use_zipcode_data:
    file = 'Zip_Zhvi_AllHomes.csv'
    data_fields, string_data = create_data_set.load_csv(file, has_field_names=True, dtype='string')
    zip_code = vec_remove_quotations(string_data[:, 1]).astype(np.int)
    state = vec_remove_quotations(string_data[:, 3])
    # year1_idx = array_functions.find_first_element(data_fields, '1996-04')
    year1_idx = array_functions.find_first_element(data_fields, '2001-01')
    # year1_idx = array_functions.find_first_element(data_fields, '2016-02')
    year2_idx = array_functions.find_first_element(data_fields, '2017-02')
    pricing_data = string_data[:, [year1_idx, year2_idx]]
    pricing_data = vec_replace(pricing_data).astype(np.float)
    zipcode_location_map = get_zipcode_locations()
    locations = np.zeros((zip_code.size, 2))
    for i, z in enumerate(zip_code):
        if z not in zipcode_location_map:
            print 'missing zipcode: ' + str(z)
            locations[i, :] = np.nan
            continue
        locations[i, :] = zipcode_location_map[z]

    all_states = np.unique(state)
else:
    if combine_with_traffic_data:
        from data_sets.taxi.process_data import load_taxi_data
        centroids_file = 'C:/Users/adgress/Desktop/ca neighborhood boundaries/centroids.csv'
        file = 'C:/Users/adgress/Desktop/Neighborhood/Neighborhood_ZhvAvgRaw_AllHomes_all.csv'

        day_locs, day_values, night_locs, night_values, _ = load_taxi_data(
            num_bins=40,
            return_coords=True
        )
        pass
    else:
        file = 'C:/Users/adgress/Desktop/Neighborhood/Neighborhood_Zhvi_BottomTier_yoy.csv'
        # file = 'C:/Users/adgress/Desktop/Neighborhood/Neighborhood_ZhvAvgRaw_AllHomes_all.csv'
        # file = 'C:/Users/adgress/Desktop/Neighborhood/Neighborhood_ZhvAvg_AllHomes_all_yoy.csv'
        # file = 'C:/Users/adgress/Desktop/Neighborhood/Neighborhood_PctOfHomesIncreasingInValues_AllHomes.csv'
        # file = 'C:/Users/adgress/Desktop/Neighborhood/Neighborhood_MedianValuePerSqft_AllHomes.csv'
        centroids_file = 'C:/Users/adgress/Desktop/ZillowNeighborhoods-NY/centroids.csv'
    data_fields, string_data = create_data_set.load_csv(file, has_field_names=True, dtype='string', return_data_frame=True)
    centroids_fields, centroids_data = create_data_set.load_csv(centroids_file, has_field_names=True, dtype='string', return_data_frame=True)
    year1_idx = array_functions.find_first_element(data_fields, '2001-01')
    year2_idx = array_functions.find_first_element(data_fields, '2017-01')
    region_ids_data = np.asarray(string_data.RegionID)
    region_ids_data = region_ids_data.astype(np.int)
    region_ids_centroids = np.asarray(centroids_data.RegionID)
    region_ids_centroids = region_ids_centroids.astype(np.int)
    pricing_data = string_data.values[:, [year1_idx, year2_idx]]
    pricing_data = vec_replace(pricing_data).astype(np.float)

    #I_data = np.argsort(region_ids_data)
    I_centroids = np.argsort(region_ids_centroids)
    #r_data_sorted = region_ids_data[I_data]
    r_centroids_sorted = region_ids_centroids[I_centroids]
    #assert (r_data_sorted == r_centroids_sorted).all()
    centroid_x = np.asarray(centroids_data.X).astype(np.float)
    centroid_y = np.asarray(centroids_data.Y).astype(np.float)
    locs = np.stack((centroid_x, centroid_y),1)
    locs = locs[I_centroids, :]
    ca_pricing_data = np.zeros((centroid_x.shape[0], 2))
    has_data = array_functions.false(ca_pricing_data.shape[0])
    for i, id in enumerate(r_centroids_sorted):
        if (id == region_ids_data).sum() == 1:
            ca_pricing_data[i, :] = pricing_data[id == region_ids_data, :]
            has_data[i] = True

    locs = locs[has_data, :]
    locations = locs
    pricing_data = ca_pricing_data[has_data, :]

    I = np.isfinite(pricing_data[:, 0]) & np.isfinite(pricing_data[:, 1])
    I &= array_functions.in_range(locations[:,0], day_locs[:,0].min(), day_locs[:,0].max())
    I &= array_functions.in_range(locations[:,1], day_locs[:,1].min(), day_locs[:,1].max())
    #I &= array_functions.in_range(locations[:,0], -123, -121)
    #I &= array_functions.in_range(locations[:,1], 37, 39)
    # I &= (state == 'OR')
    # I &= (state == 'LA')
    # I &= locations[:, 1] < 41
    # I &= locations[:, 0] > -74.5
    if combine_with_traffic_data:
        value_traffic = np.log(day_values)
        I_traffic = np.isfinite(value_traffic)
        value_traffic = value_traffic[I_traffic]
        loc_traffic = day_locs[I_traffic, :]
        loc_housing = locations[I, :]
        value_housing = pricing_data[I, 1]
        value_housing /= value_housing.max()
        if apply_log:
            value_housing = np.log(value_housing)
        data = combine_data(loc_traffic, value_traffic, loc_housing, value_housing)
    else:
        I &= np.isfinite(locations[:, 0])
        data = create_transfer_data(locations, pricing_data, I, apply_log)
    print 'n: ' + str(I.sum())
    # pricing_data[:] = 1



if run_state_tests:
    m = base_configs.MethodConfigs()
    m.cv_loss_function = loss_function.MeanSquaredError()
    m.loss_function = loss_function.MeanSquaredError()
    loss = loss_function.MeanSquaredError()
    m.use_validation = True
    m.target_labels = np.asarray([1])
    m.source_labels = np.asarray([0])
    stacking_transfer = transfer_methods.StackingTransfer(deepcopy((m)))

    m.just_target = True
    target_learner = far_transfer_methods.GraphTransfer(deepcopy(m))
    m.just_target = False
    m.just_transfer = True
    source_learner = far_transfer_methods.GraphTransfer(deepcopy(m))
    num_splits = 10
    errors = np.zeros((all_states.size, 3))
    for state_idx, s in enumerate(all_states):
        I_s = I & (state == s)
        if I_s.sum() < 100:
            print 'skipping ' + s
            continue
        data = create_transfer_data(locations, pricing_data, I_s, apply_log)
        data_splitter = DataSplitter()
        data_splitter.data = data
        splits = data_splitter.generate_splits(data.y, is_regression=True)
        split_data = data_lib.SplitData(
            data,
            splits
        )
        target_errors = np.zeros(num_splits)
        source_errors = np.zeros(num_splits)
        stacking_errors = np.zeros(num_splits)
        for split_idx in range(num_splits):
            data_copy = split_data.get_split(split_idx, num_labeled=20)

            target_results = target_learner.train_and_test(data_copy).prediction
            source_results = source_learner.train_and_test(data_copy).prediction
            stacking_results = stacking_transfer.train_and_test(data_copy).prediction
            target_errors[split_idx] = loss.compute_score(target_results)
            source_errors[split_idx] = loss.compute_score(source_results)
            stacking_errors[split_idx] = loss.compute_score(stacking_results)
        errors[state_idx, 0] = target_errors.mean()
        errors[state_idx, 1] = source_errors.mean()
        errors[state_idx, 2] = stacking_errors.mean()
        target_relative = (source_errors.mean() - target_errors.mean())/target_errors.mean()
        source_relative = (stacking_errors.mean() - target_errors.mean()) / stacking_errors.mean()
        print 'Target ' + s + ': '  + str(target_relative)
        print 'Stacking ' + s + ': ' + str(source_relative)
        if source_relative > target_relative:
            print '!!!'
    exit()

for i in range(2):
    data.x[:,i] = array_functions.normalize(data.x[:,i])
print 'n: ' + str(data.n)
if viz:
    I1 = data.data_set_ids == 0
    I2 = data.data_set_ids == 1
    fig1 = pl.figure(3)
    array_functions.plot_heatmap(data.x[I1, :], data.y[I1], sizes=30, alpha=1, subtract_min=True, fig=fig1)
    fig2 = pl.figure(4)
    array_functions.plot_heatmap(data.x[I2, :], data.y[I2], sizes=30, alpha=1, subtract_min=True, fig=fig2)
    array_functions.move_fig(fig1, 500, 500, 2000, 100)
    array_functions.move_fig(fig2, 500, 500, 2600, 100)
    pl.show(block=True)
else:
    s = 'raw_data.pkl'
    if combine_with_traffic_data:
        s = '../zillow-traffic/' + s
    helper_functions.save_object(s, data)