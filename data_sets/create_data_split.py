__author__ = 'Aubrey'


import create_data_set
from data import data as data_lib
from configs import base_configs as configs_lib
from utility import helper_functions
import numpy as np
import math
import os
from sklearn import cross_validation


def ng_configs():
    c = configs_lib.DataProcessingConfigs()
    c.is_regression = False
    return c

def boston_housing_configs():
    c = configs_lib.DataProcessingConfigs()
    c.is_regression = True
    return c

def split_data(file, configs):
    data = helper_functions.load_object(file)
    splitter = DataSplitter()
    splitData = data_lib.SplitData()
    splitData.data = data
    num_splits = 30
    perc_train = .8
    splitData.splits = splitter.generate_splits(data.y,num_splits,perc_train,data.is_regression)
    split_dir = os.path.dirname(file)
    save_file = split_dir + '/split_data.pkl'
    helper_functions.save_object(save_file,splitData)

class DataSplitter(object):
    def __init__(self):
        pass

    def generate_splits(self,y,num_splits=30,perc_train=.8,is_regression=False):
        assert y.ndim == 1
        n = len(y)
        assert np.all(~np.isnan(y))
        num_train = math.ceil(perc_train*n)
        if is_regression:
            split = cross_validation.ShuffleSplit(n,num_splits,1-perc_train)
        else:
            split = cross_validation.StratifiedShuffleSplit(y,num_splits,1-perc_train)
        splits = []
        for train,test in split:
            s = data_lib.Split(n)
            s.is_train[train] = True
            s.is_train[test] = False
            s.permutation = np.random.permutation(n)
            splits.append(s)

        return splits

if __name__ == '__main__':
    #split_data(create_data_set.boston_housing_raw_data_file, boston_housing_configs())
    split_data(create_data_set.ng_raw_data_file, ng_configs())