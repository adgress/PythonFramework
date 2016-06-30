from data import data as data_lib
from sklearn.preprocessing import LabelEncoder
import numpy as np

class NanLabelEncoding(LabelEncoder):
    def fit(self, y):
        I = ~np.isnan(y)
        super(NanLabelEncoding, self).fit(y[I])

    def fit_transform(self, y):
        I = ~np.isnan(y)
        y2 = super(NanLabelEncoding, self).fit_transform(y[I])
        y = np.copy(y)
        y[I] = y2
        return y

    def inverse_transform(self, y):
        I = ~np.isnan(y)
        y2 = super(NanLabelEncoding, self).inverse_transform(y[I])
        y = np.copy(y)
        y[I] = y2
        return y

    def transform(self, y):
        I = ~np.isnan(y)
        y2 = super(NanLabelEncoding, self).transform(y[I])
        y = np.copy(y)
        y[I] = y2
        return y

class IdentityPreprocessor(object):
    def __init__(self):
        pass

    def preprocess(self, data, configs):
        return data

class TargetOnlyPreprocessor(IdentityPreprocessor):
    def __init__(self):
        pass

    def preprocess(self, data, configs):
        target_labels = configs.target_labels
        assert target_labels.size == 1
        target_id = target_labels[0]
        data_set_ids = data.data_set_ids
        is_target_data = data_set_ids == target_id
        target_data = data.get_subset(is_target_data)
        return target_data

