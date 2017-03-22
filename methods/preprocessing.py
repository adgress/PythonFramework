from data import data as data_lib
from sklearn.preprocessing import LabelEncoder, LabelBinarizer,PolynomialFeatures
import numpy as np
import warnings

class NanLabelBinarizer(LabelBinarizer):
    def fit(self, y):
        I = ~np.isnan(y)
        super(NanLabelBinarizer, self).fit(y[I])

    def fit_transform(self, y):
        self.fit(y)
        return self.transform(y)

    def inverse_transform(self, y):
        I = ~np.isnan(y)
        y2 = super(NanLabelBinarizer, self).inverse_transform(y[I])
        y = np.copy(y)
        y[I] = y2
        return y

    def transform(self, y):
        I = ~np.isnan(y)
        y2 = super(NanLabelBinarizer, self).transform(y[I])
        y = np.copy(y)
        y[I] = y2
        return y

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

    def prefix(self):
        return 'Identity'

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

    def prefix(self):
        return 'TargetOnly'

class SelectSourcePreprocessor(IdentityPreprocessor):
    def __init__(self, sources_to_keep):
        self.sources_to_keep = sources_to_keep

    def preprocess(self, data, configs):
        target_labels = configs.target_labels
        assert target_labels.size == 1
        target_id = target_labels[0]
        data_set_ids = data.data_set_ids
        assert data_set_ids.size > 0
        should_keep = data_set_ids == target_id
        for source_id in self.sources_to_keep:
            if not (source_id == data_set_ids).any():
                warnings.warn('data_set_ids doesn'' contain id: ' + str(source_id))
            should_keep |= data_set_ids == source_id
        processed_data = data.get_subset(should_keep)
        return processed_data

    def prefix(self):
        return 'Sources=' + str(self.sources_to_keep)

class BasisExpansionPreprocessor(IdentityPreprocessor):
    BASIS_IDENTITY = 0
    BASIS_QUADRATIC = 1
    BASIS_QUADRATIC_FEW = 2

    def __init__(self, basis_expansion_type=None):
        super(BasisExpansionPreprocessor, self).__init__()
        self.basis_expansion_type = basis_expansion_type
        if self.basis_expansion_type is None:
            self.basis_expansion_type = BasisExpansionPreprocessor.BASIS_IDENTITY

    def preprocess(self, data, configs):
        x_new = None
        if self.basis_expansion_type == BasisExpansionPreprocessor.BASIS_IDENTITY:
            x_new = data.x
        elif self.basis_expansion_type == BasisExpansionPreprocessor.BASIS_QUADRATIC:
            x_new = self._basis_quadratic(data.x)
        elif self.basis_expansion_type == BasisExpansionPreprocessor.BASIS_QUADRATIC_FEW:
            x_new = self._basis_quadratic_few(data.x)
        data.x = x_new
        return data

    def _basis_quadratic_few(self, x):
        n, p = x.shape
        x_new = np.zeros((n, 2*p))
        x_new[:, 0:p] = x
        for i in range(0, p):
            xi = x[:,i]**2
            x_new[:,p+i] = xi
        return x_new

    def _basis_quadratic(self, x):
        poly = PolynomialFeatures(2)
        x_new = poly.fit_transform(x)
        #First feature is a constant 1, so discard it
        x_new = x_new[:, 1:]
        return x_new

    def prefix(self):
        if self.basis_expansion_type == BasisExpansionPreprocessor.BASIS_QUADRATIC:
            return 'QuadFeats'
        elif self.basis_expansion_type == BasisExpansionPreprocessor.BASIS_QUADRATIC_FEW:
            return 'QuadFeatsFew'
        return None

class BasisQuadraticPreprocessor(BasisExpansionPreprocessor):
    def __init__(self):
        super(BasisQuadraticPreprocessor, self).__init__(BasisExpansionPreprocessor.BASIS_QUADRATIC)

class BasisQuadraticFewPreprocessor(BasisExpansionPreprocessor):
    def __init__(self):
        super(BasisQuadraticFewPreprocessor, self).__init__(BasisExpansionPreprocessor.BASIS_QUADRATIC_FEW)



















