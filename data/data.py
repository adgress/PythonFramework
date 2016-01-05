__author__ = 'Aubrey'


import abc
import numpy as np
import collections
import copy
from utility import array_functions

data_subset = collections.namedtuple('DataSubset',['x','y'])

TYPE_TARGET = 1
TYPE_SOURCE = 2

class LabeledVector(object):
    def __init__(self):
        self.y = np.empty(0)
        self.true_y = np.empty(0)
        self.is_train = np.empty(0)
        self.type = None
        self.data_set_ids = None
        self.instance_weights = None
        self.is_regression = None

    @property
    def n(self):
        return self.y.shape[0]

    @property
    def n_train(self):
        return np.sum(self.is_train)

    @property
    def n_test(self):
        return np.sum(self.is_test)

    @property
    def n_target(self):
        return self.is_target.sum()

    @property
    def n_source(self):
        return self.is_source.sum()

    @property
    def n_source_test(self):
        return (self.is_source & self.is_test).sum()

    @property
    def n_per_y_source(self):
        return self._n_per_y(self.is_source)

    @property
    def n_per_y_target(self):
        return self._n_per_y(self.is_target)

    @property
    def n_per_y(self):
        return self.n_per_y

    def _n_per_y(self, inds=None):
        if inds is None:
            inds = np.asarray(range(self.n))
        y = self.y[inds]
        d = {
            np.nan: sum(np.isnan(y))
        }
        if self.is_regression:
            d['other'] = self.is_labeled[inds].sum()
        else:
            keys = np.unique(y)
            keys = keys[~np.isnan(keys)]
            for k in keys:
                d[int(k)] = (y == k).sum()
        return d

    @property
    def is_test(self):
        return ~self.is_train

    @property
    def is_target(self):
        if self.type is None:
            return np.ones(self.y.shape)
        return self.type == TYPE_TARGET

    @property
    def is_source(self):
        if self.type is None:
            return np.zeros(self.y.shape)
        return self.type == TYPE_SOURCE

    @property
    def y_labeled(self):
        return self.y[self.is_labeled]

    @property
    def classes(self):
        return np.unique(self.y_labeled)
    @property
    def is_labeled(self):
        return ~np.isnan(self.y)

    def reveal_labels(self, inds):
        self.y[inds] = self.true_y[inds]

    #Note: This changes both y AND true_y
    def add_noise(self, noise_rate, I=None, classes=None):
        assert not self.is_regression
        assert self.is_regression is not None
        if I is None:
            I = array_functions.true(self.n)
        if classes is None:
            I = self.classes
        to_switch = np.random.rand(self.n) <= noise_rate
        to_switch = to_switch.nonzero()[0]
        for i in to_switch:
            if not I[i]:
                continue
            old_y = self.y[i]
            y_ind = classes == old_y
            assert any(classes)
            p = np.ones(len(classes)) * 1.0 / (len(classes)-1)
            p[classes == old_y] = 0
            new_y = np.random.choice(classes,p=p)
            self.y[i] = new_y
            self.true_y[i] = new_y


class LabeledData(LabeledVector):
    def __init__(self):
        super(LabeledData, self).__init__()
        self.is_regression = None

    def repair_data(self):
        if self.type is None or len(self.type) == 0:
            self.type = TYPE_TARGET*np.ones(self.y.shape)
        self.is_train = self.is_train.astype(bool)
        pass

    @property
    def n_train_labeled(self):
        return np.sum(self.is_labeled & self.is_train)



    def get_subset(self,to_select):
        d = self.__dict__
        new_data = Data()
        for key,value in d.items():
            if array_functions.is_matrix(value) and value.shape[0] == self.n:
                setattr(new_data,key,value[to_select])
            elif hasattr(value,'deepcopy'):
                setattr(new_data,key,value.deepcopy())
            else:
                setattr(new_data,key,value)
        return new_data

    def get_transfer_inds(self,labels_or_ids):
        if self.is_regression:
            return array_functions.find_set(self.data_set_ids,labels_or_ids)
        else:
            return array_functions.find_set(self.y,labels_or_ids)

    def get_transfer_subset(self,labels_or_ids,include_unlabeled=False):
        assert len(labels_or_ids) > 0
        if self.is_regression:
            inds = array_functions.find_set(self.data_set_ids,labels_or_ids)
            if not include_unlabeled:
                inds = inds & self.is_labeled
        else:
            inds = array_functions.find_set(self.y,labels_or_ids)
            if include_unlabeled:
                inds = inds | ~self.is_labeled
        return self.get_subset(inds)

    def get_with_labels(self,labels):
        inds = array_functions.find_set(self.true_y,labels)
        return self.get_subset(inds)

    def get_data_set_ids(self, data_set_ids):
        inds = array_functions.find_set(self.data_set_ids,data_set_ids)
        return self.get_subset(inds)

    def get_with_labels_and_unlabeled(self,labels):
        inds = array_functions.find_set(self.true_y,labels) | ~self.is_labeled
        return self.get_subset(inds)

    def permute(self,permutation):
        d = self.__dict__
        for key,value in d.items():
            if array_functions.is_matrix(value):
                if value.ndim == 1:
                    value = value[permutation]
                elif value.ndim == 2:
                    value = value[permutation,:]
                else:
                    assert False
                setattr(self,key,value)

    def apply_split(self,split):
        self.is_train = split.is_train
        #self.type = split.type
        self.permute(split.permutation)

    def set_defaults(self):
        self.set_train()
        self.set_target()
        self.set_true_y()

    def set_train(self):
        self.is_train = array_functions.true(self.n)

    def set_target(self):
        self.type = TYPE_TARGET*np.ones(self.n)
        self.data_set_ids = np.zeros(self.n)

    def set_true_y(self):
        self.true_y = self.y.copy()

    def change_labels(self, curr_labels, new_labels):
        #assert len(curr_labels) == len(new_labels)
        assert curr_labels.shape[1] == new_labels.shape[0]
        if curr_labels.ndim == 1:
            curr_labels = np.expand_dims(curr_labels,1)
        new_y = self.y.copy()
        new_true_y = self.true_y.copy()
        for i in range(curr_labels.shape[0]):
            l = curr_labels[i,:]
            for curr, new in zip(l, new_labels):
                new_y[self.y == curr] = new
                new_true_y[self.true_y == curr] = new
        self.y = new_y
        self.true_y = new_true_y

    def remove_test_labels(self):
        self.y[self.is_test] = np.nan

    def rand_sample(self,perc=.1):
        p = np.random.permutation(self.n)
        m = np.ceil(perc*self.n)
        to_use = p[:m]
        return self.get_subset(to_use)


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

    def arg_sort(self):
        assert self.x.shape[1] == 1
        x = np.squeeze(self.x)
        return np.squeeze(self.x.argsort(0))

class Split(object):
    def __init__(self,n=0):
        self.permutation = np.empty(n)
        self.is_train = array_functions.true(n)
        #self.type = np.full(n,TYPE_TARGET)

class SplitData(object):
    def __init__(self,data=Data(),splits=[]):
        self.data = data
        self.splits = splits
        self.labels_to_keep = None
        self.labels_to_not_sample = {}
        self.target_labels = None
        self.use_data_set_ids = True

    def get_split(self, i, num_labeled=None):
        if 'use_data_set_ids' not in self.__dict__:
            self.use_data_set_ids = True
        d = copy.deepcopy(self.data)
        split = self.splits[i]
        d.apply_split(split)
        if self.labels_to_keep is not None:
            #d = d.get_with_labels(self.labels_to_keep)
            d = d.get_transfer_subset(self.labels_to_keep)
        if num_labeled is not None:
            if d.is_regression:
                labeled_inds = np.nonzero(d.is_train)[0]
                if self.use_data_set_ids and \
                                d.data_set_ids is not None and \
                                self.target_labels is not None:
                    labeled_inds = np.nonzero(d.is_train & (d.data_set_ids == self.target_labels))[0]
                to_clear = labeled_inds[num_labeled:]
                d.y[to_clear] = np.nan
            else:
                d.y = d.y.astype('float32')
                d.true_y = d.true_y.astype('float32')
                classes = d.classes
                for c in classes:
                    if c in self.labels_to_not_sample:
                        continue
                    class_inds_train = np.nonzero((d.y==c) & d.is_train)[0]
                    assert len(class_inds_train) >= num_labeled
                    d.y[class_inds_train[num_labeled:]] = np.nan
                    class_inds_test = np.nonzero((d.y==c) & ~d.is_train)[0]
                    d.y[class_inds_test] = np.nan
        return d
