__author__ = 'Aubrey'


import create_data_set
from data import data as data_lib
from configs import base_configs as configs_lib
from utility import helper_functions
from utility import array_functions
import numpy as np
import math
import os
from sklearn import cross_validation

def regression_configs():
    c = configs_lib.DataProcessingConfigs()
    c.is_regression = True
    return c

def classification_configs():
    c = configs_lib.DataProcessingConfigs()
    c.is_regression = False
    return c

def hypothesis_transfer_configs(source_ids):
    c = configs_lib.DataProcessingConfigs()
    c.is_regression = False
    c.split_data_set_ids = np.zeros(1)
    c.data_set_ids_to_keep = np.asarray(source_ids)
    return c

def synthetic_classification_configs():
    c = configs_lib.DataProcessingConfigs()
    c.is_regression = False
    return c

def ng_configs():
    c = configs_lib.DataProcessingConfigs()
    c.is_regression = False
    return c

def boston_housing_configs():
    c = configs_lib.DataProcessingConfigs()
    c.is_regression = True
    return c

def synthetic_step_transfer_configs():
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
    keep_for_splitting = None
    if configs.split_data_set_ids is not None:
        keep_for_splitting = array_functions.false(data.n)
        keep_for_splitting[data.data_set_ids == 0] = True
    #Pretend data_set_ids is a label vector to ensure each data set is split equally
    if data.is_regression and data.data_set_ids is not None:
        assert len(data.data_set_ids) == data.n
        is_regression = False
        splitData.splits = splitter.generate_splits(
            data.data_set_ids,
            num_splits,
            perc_train,
            is_regression,
            keep_for_splitting
        )
    else:
        splitData.splits = splitter.generate_splits(
            data.y,
            num_splits,
            perc_train,
            data.is_regression,
            keep_for_splitting
        )
    splitData.data_set_ids_to_keep = configs.data_set_ids_to_keep
    split_dir = os.path.dirname(file)
    save_file = split_dir + '/split_data.pkl'
    helper_functions.save_object(save_file,splitData)
    return splitData

class DataSplitter(object):
    def __init__(self):
        pass

    def generate_identity_split(self, is_train):
        s = data_lib.Split(is_train.shape[0])
        s.is_train = is_train
        s.permutation = np.asarray(range(is_train.shape[0]))
        return [s]

    def generate_splits(self,y,num_splits=30,perc_train=.8,is_regression=False,keep_for_splitting=None):
        assert y.ndim == 1
        keep_in_train_set = array_functions.false(len(y))
        if keep_for_splitting is not None and len(keep_for_splitting) > 0:
            keep_in_train_set[~keep_for_splitting] = True
            #keep_in_train_set[~array_functions.to_boolean(keep_for_splitting)] = True
        is_labeled = ~np.isnan(y)
        keep_in_train_set[~is_labeled] = True

        n = len(y)
        #if keep_for_splitting is not None:
         #   y_for_split = y[keep_for_splitting]
          #  target_inds = keep_for_splitting.nonzero()[0]
        y_for_split = y[~keep_in_train_set]
        n_for_split = len(y_for_split)
        inds_for_splitting = (~keep_in_train_set).nonzero()[0]
        random_state = None
        if is_regression:
            split = cross_validation.ShuffleSplit(n_for_split,num_splits,1-perc_train,random_state=random_state)
        else:
            split = cross_validation.StratifiedShuffleSplit(y_for_split,num_splits,1-perc_train,random_state=random_state)
        splits = []
        for train,test in split:
            s = data_lib.Split(n)
            s.is_train[:] = True
            s.is_train[inds_for_splitting[test]] = False
            '''
            if keep_for_splitting is not None:
                s.is_train[:] = True
                s.is_train[target_inds[test]] = False
            else:
                s.is_train[train] = True
                s.is_train[test] = False
            '''
            s.permutation = np.random.permutation(n)
            splits.append(s)

        return splits

def run_main():
    #split_data('synthetic_linear_reg500-50-1/raw_data.pkl', boston_housing_configs())
    #split_data(create_data_set.adience_aligned_cnn_1_per_instance_id_file, regression_configs())
    #split_data(create_data_set.wine_file % '-red', regression_configs())
    #split_data(create_data_set.concrete_file % '', regression_configs())
    #split_data(create_data_set.drosophila_file, regression_configs())
    '''
    s = split_data('synthetic_hyp_trans_class500-50-1.0-0.3-2-2/raw_data.pkl',
               hypothesis_transfer_configs([1,2,3,4])
    )
    '''
    #split_data('synthetic_linear_reg500-10-1-nnz=4/raw_data.pkl', regression_configs())
    #split_data('synthetic_linear_reg500-50-1.01/raw_data.pkl', regression_configs())
    #split_data(create_data_set.kc_housing_file, regression_configs())
    #split_data('synthetic_linear_reg500-10-1.01/raw_data.pkl', regression_configs())
    #split_data('synthetic_flip/raw_data.pkl', regression_configs())
    split_data('pollution-2-800/raw_data.pkl', regression_configs())
    pass

if __name__ == '__main__':
    run_main()