__author__ = 'Aubrey and Evan'

import numpy as np
from sklearn import metrics
import math
import abc

class LossFunction:
    """
    Implements loss functions
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self.name = None
        self.short_name = None


    def compute_score(self, y1, y2, I=None):
        if I is not None:
            y1 = y1[I]
            y2 = y2[I]
        return self._compute_score(y1,y2)

    @abc.abstractmethod
    def _compute_score(self,y1,y2):
        pass

class ZeroOneError(LossFunction):
    def __init__(self):
        self.name = 'zero_one_error'
        self.short_name = '0-1'

    def _compute_score(self, y1, y2):
        return metrics.zero_one_loss(y1, y2)
class MeanSquaredError(LossFunction):
    def __init__(self):
        self.name = 'mean_squared_error'
        self.short_name = 'MSE'

    def _compute_score(self, y1, y2):
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
        return metrics.mean_absolute_error(y1, y2)
