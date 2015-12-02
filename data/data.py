__author__ = 'Aubrey'


import abc
import numpy as np
import collections
import copy
from utility import array_functions

data_subset = collections.namedtuple('DataSubset',['x','y'])

TYPE_TARGET = 1
TYPE_SOURCE = 2

class LabeledData(object):
    def __init__(self):
        self.y = np.empty(0)
        self.true_y = np.empty(0)
        self.is_train = np.empty(0)
        self.type = None
        self.data_set_ids = None
        self.is_regression = None

    @property
    def n(self):
        return self.y.shape[0]

    @property
    def n_train(self):
        return np.sum(self.is_train)

    @property
    def n_train_labeled(self):
        return np.sum(self.is_labeled & self.is_train)

    @property
    def is_target(self):
        if self.type is None:
            return np.ones(self.y.shape)
        return self.type == TYPE_TARGET

    def is_source(self):
        if self.type is None:
            return np.zeros(self.y.shape)
        return self.type == TYPE_SOURCE

    @property
    def is_labeled(self):
        return ~np.isnan(self.y)

    @property
    def y_labeled(self):
        return self.y[self.is_labeled]

    @property
    def classes(self):
        return np.unique(self.y_labeled)

    def get_subset(self,to_select):
        d = self.__dict__
        new_data = Data()
        for key,value in d.items():
            if isinstance(value,np.ndarray):
                setattr(new_data,key,value[to_select])
            elif hasattr(value,'deepcopy'):
                setattr(new_data,key,value.deepcopy())
            else:
                setattr(new_data,key,value)
        return new_data

    def permute(self,permutation):
        d = self.__dict__
        for key,value in d.items():
            if isinstance(value,np.ndarray):
                if value.ndim == 1:
                    value = value[permutation]
                elif value.ndim == 2:
                    value = value[permutation,:]
                else:
                    assert False
                setattr(self,key,value)

    def apply_split(self,split):
        self.is_train = split.is_train
        self.type = split.type
        self.permute(split.permutation)

    def set_defaults(self):
        self.set_train()
        self.set_target()
        self.set_true_y()

    def set_train(self):
        self.is_train = np.ones((self.n))

    def set_target(self):
        self.is_train = TYPE_TARGET*np.ones(self.n)

    def set_true_y(self):
        self.true_y = self.y

class Data(LabeledData):
    def __init__(self):
        super(Data, self).__init__()
        self.x = np.empty((0,0))
        self.y = np.empty(0)
        self.feature_names = None
        self.label_names = None

    @property
    def p(self):
        return self.x.shape[1]

    def labeled_training_data(self):
        I = self.is_train & self.is_labeled
        return data_subset(x=self.x[I,:],y=self.y[I])

    def labeled_test_data(self):
        I = ~self.is_train & self.is_labeled
        return data_subset(x=self.x[I,:],y=self.y[I])

class Split(object):
    def __init__(self,n=0):
        self.permutation = np.empty(n)
        self.is_train = array_functions.true(n)
        self.type = np.full(n,TYPE_TARGET)

class SplitData(object):
    def __init__(self,data=Data(),splits=[]):
        self.data = data
        self.splits = splits

    def get_split(self, i, num_labeled=None):
        d = copy.deepcopy(self.data)
        split = self.splits[i]
        d.apply_split(split)
        if num_labeled is not None:
            if d.is_regression:
                labeled_inds = np.nonzero(d.is_train)[0]
                to_clear = labeled_inds[num_labeled:]
                d.y[to_clear] = np.nan
            else:
                classes = d.classes()
                for c in classes:
                    pass
                assert False
        return d
