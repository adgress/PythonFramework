__author__ = 'Aubrey'

import abc
from saveable.saveable import Saveable
from configs.base_configs import MethodConfigs
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn import dummy
import numpy as np
from results_class.results import Output
import collections


class Method(Saveable):

    def __init__(self,configs=MethodConfigs()):
        super(Method, self).__init__(configs)
        self.configs = configs
        self._params = []
        self.cv_params = {}
        self.is_classifier = True

    @property
    def params(self):
        return self._params

    def set_params(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    @abc.abstractmethod
    def train_and_test(self, data):
        pass

    def run_method(self, data):
        self.train(data)
        return self.predict(data)

    @abc.abstractmethod
    def train(self, data):
        pass

    @abc.abstractmethod
    def predict(self, data):
        pass

class ScikitLearnMethod(Method):

    _short_name_dict = {
        'RidgeRegression': 'RidgeReg',
        'DummyClassifier': 'DumClass'
    }

    def __init__(self,configs=MethodConfigs(),skl_method=None):
        super(ScikitLearnMethod, self).__init__(configs)
        self.skl_method = skl_method

    def train(self, data):
        labeled_train = data.labeled_training_data()
        self.skl_method.fit(labeled_train.x, labeled_train.y)

    def predict(self, data):
        o = Output(data)
        o.y = self.skl_method.predict(data.x)
        return o

    def set_params(self, **kwargs):
        super(ScikitLearnMethod,self).set_params(**kwargs)
        self.skl_method.set_params(**kwargs)

    def _skl_method_name(self):
        return repr(self.skl_method).split('(')[0]

    @property
    def prefix(self):
        return "SKL-" + ScikitLearnMethod._short_name_dict[self._skl_method_name()]

class SKLRidgeRegression(ScikitLearnMethod):
    def __init__(self,configs=None):
        super(SKLRidgeRegression, self).__init__(configs, linear_model.Ridge())
        self.cv_params['alpha'] = 10**np.asarray(([range(-5,5)]))
        self.set_params(alpha=0,fit_intercept=True,normalize=True)

class SKLLogisticRegression(ScikitLearnMethod):
    def __init__(self,configs=None):
        super(SKLLogisticRegression, self).__init__(configs, linear_model.LogisticRegression())
        self.cv_params['C'] = 10**np.asarray(list(reversed([range(-5, 5)])))
        self.set_params(C=0,fit_intercept=True,penalty='l2')

class SKLGuessClassifier(ScikitLearnMethod):
    def __init__(self,configs=None):
        super(dummy.DummyClassifier('uniform'), self).__init__(configs)

class SKLMeanRegressor(ScikitLearnMethod):
    def __init__(self,configs=None):
        super(dummy.DummyRegressor('mean'), self).__init__(configs)