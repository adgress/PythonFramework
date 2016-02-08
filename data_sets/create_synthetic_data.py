from data import data as data_class
from utility import helper_functions
from utility import array_functions
import scipy
import numpy as np
import pandas as pd
import math
from PyMTL_master.src import PyMTL
from viz_data import viz_features

synthetic_dim = 5

synthetic_step_transfer_file = 'synthetic_step_transfer/raw_data.pkl'
synthetic_delta_linear_file = 'synthetic_delta_linear_transfer/raw_data.pkl'
synthetic_cross_file = 'synthetic_cross_transfer/raw_data.pkl'
synthetic_step_kd_transfer_file = 'synthetic_step_transfer_%d/raw_data.pkl'
synthetic_step_linear_transfer_file = 'synthetic_step_linear_transfer/raw_data.pkl'
synthetic_classification_file = 'synthetic_classification/raw_data.pkl'
synthetic_classification_local_file = 'synthetic_classification_local/raw_data.pkl'
synthetic_slant_file = 'synthetic_slant/raw_data.pkl'
synthetic_curve_file = 'synthetic_curve/raw_data.pkl'

def create_synthetic_classification(file_dir='',local=True):
    dim = 1
    n_target = 200
    n_source = 200
    n = n_target + n_source
    data = data_class.Data()
    data.x = np.random.uniform(0,1,(n,dim))
    data.data_set_ids = np.zeros(n)
    data.data_set_ids[n_target:] = 1
    data.y = np.zeros(n)
    x, ids = data.x, data.data_set_ids
    I = array_functions.in_range(x,0,.25)
    I2 = array_functions.in_range(x,.25,.5)
    I3 = array_functions.in_range(x,.5,.75)
    I4 = array_functions.in_range(x,.75,1)
    id0 = ids == 0
    id1 = ids == 1
    data.y[I & id0] = 1
    data.y[I2 & id0] = 2
    data.y[I3 & id0] = 1
    data.y[I4 & id0] = 2

    data.y[I & id1] = 3
    data.y[I2 & id1] = 4
    data.y[I3 & id1] = 3
    data.y[I4 & id1] = 4
    if local:
        data.y[I3 & id1] = 4
        data.y[I4 & id1] = 3
    data.set_true_y()
    data.set_train()
    data.is_regression = False
    noise_rate = 0
    #data.add_noise(noise_rate)
    data.add_noise(noise_rate, id0, np.asarray([1,2]))
    data.add_noise(noise_rate, id1, np.asarray([3,4]))
    s = synthetic_classification_file
    if local:
        s = synthetic_classification_local_file
    i = id1
    array_functions.plot_2d(data.x[i,:],data.y[i])
    if file_dir != '':
        s = file_dir + '/' + s
    helper_functions.save_object(s,data)

def create_synthetic_step_transfer(file_dir='', dim=1):
    n_target = 100
    n_source = 100
    n = n_target + n_source
    sigma = .5
    data = data_class.Data()
    data.x = np.random.uniform(0,1,(n,dim))
    data.data_set_ids = np.zeros(n)
    data.data_set_ids[n_target:] = 1
    data.y = np.zeros(n)
    data.y[(data.data_set_ids == 0) & (data.x[:,0] >= .5)] = 2
    data.y += np.random.normal(0,sigma,n)
    data.set_defaults()
    data.is_regression = True
    if dim == 1:
        array_functions.plot_2d(data.x,data.y,data.data_set_ids)
    s = synthetic_step_transfer_file
    if dim > 1:
        s = synthetic_step_kd_transfer_file % dim
    if file_dir != '':
        s = file_dir + '/' + s
    helper_functions.save_object(s,data)


def create_synthetic_step_linear_transfer(file_dir=''):
    n_target = 100
    n_source = 100
    n = n_target + n_source
    sigma = .5
    data = data_class.Data()
    data.x = np.random.uniform(0,1,(n,1))
    data.data_set_ids = np.zeros(n)
    data.data_set_ids[n_target:] = 1
    data.y = np.reshape(data.x*5,data.x.shape[0])
    data.y[(data.data_set_ids == 1) & (data.x[:,0] >= .5)] += 4
    data.y += np.random.normal(0,sigma,n)
    data.set_defaults()
    data.is_regression = True
    array_functions.plot_2d(data.x,data.y,data.data_set_ids,title='Linear Step Data Set')
    s = synthetic_step_linear_transfer_file
    if file_dir != '':
        s = file_dir + '/' + s
    helper_functions.save_object(s,data)

def create_synthetic_delta_linear_transfer():
    slope = 5
    target_fun = lambda x: slope*x
    source_fun = lambda x: slope*x + 4
    data = create_synthetic_regression_transfer(target_fun, source_fun)
    array_functions.plot_2d(data.x,data.y,data.data_set_ids,title='Linear Delta Data Set')
    s = synthetic_delta_linear_file
    helper_functions.save_object(s, data)

def create_synthetic_cross_transfer():
    slope = 5
    target_fun = lambda x: slope*x
    source_fun = lambda x: -slope*x + 5
    data = create_synthetic_regression_transfer(target_fun, source_fun)
    s = synthetic_cross_file
    helper_functions.save_object(s, data)

def create_synthetic_slant_transfer():
    target_fun = lambda x: 2*x
    source_fun = lambda x: 2.5*x + 1
    data = create_synthetic_regression_transfer(target_fun, source_fun)
    s = synthetic_slant_file
    helper_functions.save_object(s, data)

def create_synthetic_curve_transfer():
    target_fun = lambda x: x**2
    source_fun = lambda x: x**2.5 + 1
    data = create_synthetic_regression_transfer(target_fun, source_fun)
    s = synthetic_curve_file
    helper_functions.save_object(s, data)

def create_synthetic_regression_transfer(target_fun, source_fun, n_target=100, n_source=100, sigma=.5):
    n = n_target + n_source
    data = data_class.Data()
    data.x = np.random.uniform(0,1,(n,1))
    data.data_set_ids = np.zeros(n)
    data.data_set_ids[n_target:] = 1
    is_target = data.data_set_ids == 0
    data.y = np.zeros(n)
    data.y[is_target] = target_fun(data.x[is_target])
    data.y[~is_target] = source_fun(data.x[~is_target])
    data.y += np.random.normal(0,sigma,n)
    data.set_true_y()
    data.set_train()
    data.is_regression = True
    return data

if __name__ == '__main__':
    create_synthetic_delta_linear_transfer()