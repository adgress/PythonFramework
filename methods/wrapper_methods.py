from data import data as data_lib
import numpy as np
from configs import base_configs
from methods import method
from copy import deepcopy
from utility import array_functions
import sklearn
from sklearn.decomposition import PCA

class PipelineMethod(method.Method):
    def __init__(self, configs=base_configs.MethodConfigs()):
        super(PipelineMethod, self).__init__(configs)
        self.preprocessing_pipeline = []
        self.base_learner = method.NadarayaWatsonMethod(deepcopy(configs))
        self.in_train_and_test = False

    def train_and_test(self, data):
        self.in_train_and_test = True
        for l in self.preprocessing_pipeline:
            data = l.fit_and_transform(data)
        vals = self.base_learner.train_and_test(data)
        self.in_train_and_test = False
        return vals

    def predict(self, data):
        if not self.in_train_and_test:
            for l in self.preprocessing_pipeline:
                data = l.transform(data)
        return self.base_learner.predict(data)

    def train(self, data):
        if not self.in_train_and_test:
            for l in self.preprocessing_pipeline:
                data = l.fit_and_transform(data)
        return self.base_learner.train(data)

    @property
    def prefix(self):
        s = 'Pipeline'
        for l in self.preprocessing_pipeline:
            s2 = l.prefix
            if s2 != '':
                s += '_' + s2
        s += '+' + self.base_learner.prefix
        return s


class PipelineElement(object):
    def __init__(self):
        super(PipelineElement, self).__init__()

    def fit(self, data):
        pass

    def transform(self, data):
        return data

    def fit_and_transform(self, data):
        self.fit(data)
        return self.transform(data)

    @property
    def prefix(self):
        return ''

def select_all(data):
    return array_functions.true(data.n)

def select_classes(data, classes):
    return array_functions.find_set(data.true_y, classes)

def select_subset_using_property(data, property):
    return getattr(data, property)

def selection_subset_ids(data, ids):
    return array_functions.find_set(data.data_set_ids, ids)

class PipelineSelectSubset(PipelineElement):
    def __init__(self):
        super(PipelineSelectSubset, self).__init__()
        self.selection_func = select_all

    def transform(self, data):
        I = self.selection_func(data)
        return data.get_subset(I)

    @property
    def prefix(self):
        return ''

class PipelineSelectDataIDs(PipelineSelectSubset):
    def __init__(self, ids=[0]):
        super(PipelineSelectSubset, self).__init__()
        #self.selection_func = lambda data: selection_subset_ids(data, ids)
        self.ids = ids

    def transform(self, data):
        I = selection_subset_ids(data, self.ids)
        return data.get_subset(I)

    @property
    def prefix(self):
        return ''

class PipelineSelectClasses(PipelineSelectSubset):
    def __init__(self, classes=[0, 1]):
        super(PipelineSelectSubset, self).__init__()
        #self.selection_func = lambda data: select_classes(data, classes)
        self.classes = classes

    def transform(self, data):
        I = select_classes(data, self.classes)
        return data.get_subset(I)

    @property
    def prefix(self):
        return ''

class PipelineSelectLabeled(PipelineSelectSubset):
    def __init__(self):
        super(PipelineSelectLabeled, self).__init__()
        #self.selection_func = lambda data: select_subset_using_property(data, 'is_labeled')

    def transform(self, data):
        I = select_subset_using_property(data, 'is_labeled')
        return data.get_subset(I)

class PipelineSampleClasses(PipelineSelectSubset):
    def __init__(self, num_to_select=None):
        super(PipelineSampleClasses, self).__init__()
        #self.selection_func = lambda data: select_subset_using_property(data, 'is_labeled')
        self.num_to_select = num_to_select

    def transform(self, data):
        assert not data.is_regression
        y = data.classes
        to_keep = array_functions.true(data.n)
        if self.num_to_select is not None:
            for yi, perc in zip(y, self.num_to_select):
                inds = (data.y == yi).nonzero()[0]
                n_to_keep = np.ceil(inds.size * perc)
                to_keep[inds[n_to_keep:]] = False
            data = data.get_subset(to_keep)
        return data


    @property
    def prefix(self):
        s = ''
        if self.num_to_select is not None:
            s += 'SampleClasses=' + str(self.num_to_select)
        return s

class PiplineUseVariance(PipelineElement):
    def __init__(self, use_variance=True):
        super(PiplineUseVariance, self).__init__()
        self.base_learner = method.NadarayaWatsonMethod()
        self.use_variance = use_variance

    def fit(self, data):
        if self.use_variance:
            self.base_learner.train_and_test(data)

    def transform(self, data):
        if self.use_variance:
            data = deepcopy(data)
            data.true_y = np.abs(self.base_learner.predict(data).y - data.true_y)
            data.y[data.is_labeled] = data.true_y[data.is_labeled]
        return data

    @property
    def prefix(self):
        s = ''
        if self.use_variance:
            s = 'useVar'
        return s

class PipelineSKLTransform(PipelineElement):
    def __init__(self, skl_transform=None):
        super(PipelineSKLTransform, self).__init__()
        self.skl_transform = skl_transform

    def fit(self, data):
        self.skl_transform.fit(data.x)

    def transform(self, data):
        data.x = self.skl_transform.transform(data.x)
        return data

    @property
    def prefix(self):
        return ''

class PipelineMakeRegression(PipelineElement):
    def __init__(self):
        super(PipelineMakeRegression, self).__init__()

    def transform(self, data):
        data.is_regression = True
        classes = np.sort(np.unique(data.true_y))
        assert classes.size == 2
        data.change_labels(classes, [0, 1])
        return data

class PipelineChangeClasses(PipelineElement):
    def __init__(self, classes=dict(), save_y_orig=True):
        super(PipelineChangeClasses, self).__init__()
        self.classes = classes
        self.save_y_orig = save_y_orig

    def transform(self, data):
        if self.save_y_orig:
            data.y_orig = data.true_y.copy()
        data.change_labels_dict(self.classes)
        return data

class PipelineAddClusterNoise(PipelineElement):
    def __init__(self, n_per_cluster=10, num_clusters=1, flip_labels=True, y_offset=5, save_y_orig=False):
        super(PipelineAddClusterNoise, self).__init__()
        self.n_per_cluster = n_per_cluster
        self.num_clusters = num_clusters
        self.flip_labels = flip_labels
        self.y_offset = y_offset
        self.save_y_orig = save_y_orig

    def transform(self, data):
        should_add_noise = array_functions.false(data.n)
        for i in range(self.num_clusters):
            idx = np.random.choice(data.n)
            cluster_inds = array_functions.find_knn(data.x, data.x[idx], k=self.n_per_cluster)
            should_add_noise[cluster_inds] = True
        if self.save_y_orig:
            data.y_orig = data.true_y.copy()
        if should_add_noise.any():
            if self.flip_labels:
                data.flip_label(should_add_noise)
            else:
                data.true_y[should_add_noise] += self.y_offset
                data.y[should_add_noise] += self.y_offset
        data.is_noisy = should_add_noise
        return data

class IdentityWrapper(method.Method):
    def __init__(self, configs=base_configs.MethodConfigs()):
        super(IdentityWrapper, self).__init__(configs)
        self.base_learner = method.NadarayaWatsonMethod(deepcopy(configs))

    def train_and_test(self, data):
        return self.base_learner.train_and_test(data)

    def predict(self, data):
        return self.base_learner.predict(data)

    def train(self, data):
        return self.base_learner.train(data)

    @property
    def prefix(self):
        return 'IdentityWrapper+' + self.base_learner.prefix

class TargetOnlyWrapper(IdentityWrapper):
    def __init__(self, configs=base_configs.MethodConfigs()):
        super(TargetOnlyWrapper, self).__init__(configs)

    def train_and_test(self, data):
        data = data.get_transfer_subset(self.configs.target_labels, include_unlabeled=True)
        return self.base_learner.train_and_test(data)

    @property
    def prefix(self):
        return 'TargetOnlyWrapper+' + self.base_learner.prefix

class TransformWrapper(IdentityWrapper):
    def __init__(self, configs=base_configs.MethodConfigs()):
        super(TransformWrapper, self).__init__(configs)
        self.transform = PCA(n_components=2)

    def train_and_test(self, data):
        data = deepcopy(data)
        data.x = self.transform.fit_transform(data.x)
        return self.base_learner.train_and_test(data)

    def predict(self, data):
        data = deepcopy(data)
        data.x = self.transform.transform(data.x)
        return self.base_learner.predict(data)

    def train(self, data):
        data = deepcopy(data)
        data.x = self.transform.transform(data.x)
        return self.base_learner.train(data)

    @property
    def prefix(self):
        return 'PCAWrapper+' + self.base_learner.prefix