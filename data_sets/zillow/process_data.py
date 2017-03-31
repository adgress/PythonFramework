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
apply_log = True

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


file = 'Zip_Zhvi_AllHomes.csv'
data_fields, string_data = create_data_set.load_csv(file, has_field_names=True,dtype='string')
zip_code = vec_remove_quotations(string_data[:, 1]).astype(np.int)
state = vec_remove_quotations(string_data[:,3])
#year1_idx = array_functions.find_first_element(data_fields, '1996-04')
year1_idx = array_functions.find_first_element(data_fields, '2016-02')
year2_idx = array_functions.find_first_element(data_fields, '2017-02')
pricing_data =  string_data[:, [year1_idx, year2_idx]]
pricing_data = vec_replace(pricing_data).astype(np.float)
zipcode_location_map = get_zipcode_locations()
locations = np.zeros((zip_code.size,2))
for i, z in enumerate(zip_code):
    if z not in zipcode_location_map:
        print 'missing zipcode: ' + str(z)
        locations[i,:] = np.nan
        continue
    locations[i,:] = zipcode_location_map[z]

I = np.isfinite(year1_idx) & np.isfinite(year2_idx) & np.isfinite(locations[:,0])
all_states = np.unique(state)


run_state_tests = False

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
        print 'Target ' + s + ': '  + str((source_errors.mean() - target_errors.mean())/target_errors.mean())
        print 'Stacking ' + s + ': ' + str((stacking_errors.mean() - target_errors.mean()) / stacking_errors.mean())
    exit()
I &= (state == 'OR')
data = create_transfer_data(locations, pricing_data, I, apply_log)
viz = False
print 'n: ' + str(I.sum())
#pricing_data[:] = 1
print 'n: ' + str(data.n)
if viz:
    fig1 = pl.figure(3)
    array_functions.plot_heatmap(locations[I,:], pricing_data[I,0], sizes=30, alpha=1, subtract_min=False, fig=fig1)
    fig2 = pl.figure(4)
    array_functions.plot_heatmap(locations[I,:], pricing_data[I,1], sizes=30, alpha=1, subtract_min=False, fig=fig2)
    pl.show(block=True)
else:
    helper_functions.save_object('raw_data.pkl', data)