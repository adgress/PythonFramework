__author__ = 'Aubrey'


import abc
import numpy as np
import collections
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

    @property
    def n(self):
        return self.y.shape[1]

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

    def get_subset(self,to_select):
        d = self.__dict__
        new_data = Data()
        for key,value in d.items():
            if isinstance(value,np.ndarray):
                setattr(new_data,value[to_select])
            elif hasattr(value,'deepcopy'):
                setattr(new_data,value.deepcopy())
            else:
                setattr(new_data,value)
        return new_data


    def set_train(self):
        self.is_train = np.ones((self.n))

    def set_target(self):
        self.is_train = np.zeros(self.n)


class Data(LabeledData):
    def __init__(self):
        super(Data, self).__init__()
        self.x = np.empty((0,0))
        self.y = np.empty(0)

    def p(self):
        return self.x.shape[2]

    def labeled_training_data(self):
        I = self.is_train & self.is_labeled
        return data_subset(x=self.x[I,:],y=self.y[I])

    def labeled_test_data(self):
        I = ~self.is_train & self.is_labeled
        return data_subset(x=self.x[I,:],y=self.y[I])
