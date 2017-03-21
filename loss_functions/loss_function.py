__author__ = 'Aubrey and Evan'

import numpy as np
from sklearn import metrics
import math
import abc
from utility import array_functions
from numpy.linalg import norm

class LossFunction(object):
    """
    Implements loss functions
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self.name = None
        self.short_name = None

    def compute_score(self, output):
        y1 = np.squeeze(output.y)
        y2 = np.squeeze(output.true_y)
        I = ~output.is_train
        return self._compute_score(y1[I], y2[I])
        '''
        return loss_function.compute_score(
            self.y,
            self.true_y,
            ~self.is_train
        )
        '''
    '''
    def compute_score(self, y1, y2, I=None):
        if I is not None:
            y1 = y1[I]
            y2 = y2[I]
        return self._compute_score(y1,y2)
    '''
    @abc.abstractmethod
    def _compute_score(self,y1,y2):
        pass

class ZeroOneError(LossFunction):
    def __init__(self):
        self.name = 'zero_one_error'
        self.short_name = '0-1'

    def _compute_score(self, y1, y2):
        return metrics.zero_one_loss(y1, y2)

class LogLoss(LossFunction):
    def __init__(self):
        self.name = 'log_loss'
        self.short_name = 'LL'

    def _compute_score(self, y1, y2):
        all_zero = array_functions.is_all_zero_column(y1) & array_functions.is_all_zero_column(y2)
        column_inds = (~all_zero).nonzero()[0]
        assert column_inds.size == 2
        y1 = y1[:,column_inds]
        y2 = y2[:,column_inds]
        l = metrics.log_loss(y2, y1, eps=.1)
        return l


class MeanSquaredError(LossFunction):
    def __init__(self):
        self.name = 'mean_squared_error'
        self.short_name = 'MSE'

    def _compute_score(self, y1, y2):
        if y1.size == 0 or not np.isfinite(y1).all() or not np.isfinite(y2).all():
            return np.inf
        return metrics.mean_squared_error(y1, y2)

class RootMeanSquaredError(LossFunction):
    def __init__(self):
        self.name = 'root_mean_squared_error'
        self.short_name = 'RMSE'

    def _compute_score(self, y1, y2):
        return math.sqrt(metrics.mean_squared_error(y1, y2))

class MeanAbsoluteError(LossFunction):
    def __init__(self):
        self.name = 'mean_absolute_error'
        self.short_name = 'MAE'

    def _compute_score(self,y1,y2):
        if y1.size == 0 or not np.isfinite(y1).all() or not np.isfinite(y2).all():
            return np.inf
        return metrics.mean_absolute_error(y1, y2)

class LossFunctionParams(LossFunction):
    def __init__(self):
        self.name = None
        self.short_name = None

    def compute_score(self, output):
        w1 = output.w
        w2 = output.true_w
        return self._compute_score(w1/norm(w1), w2/norm(w2))

class LossFunctionParamsMeanAbsoluteError(LossFunctionParams):
    def __init__(self):
        self.name = 'w_mean_squared_error'
        self.short_name = 'wMAE'

    def _compute_score(self, w1, w2):
        return math.sqrt(metrics.mean_squared_error(w1, w2))