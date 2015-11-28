__author__ = 'Aubrey'

from sklearn import grid_search
from sklearn import cross_validation as cv

class CrossValidation(object):
    def __init__(self):
        self.method = None
        self.splits = None
        self.cv_params = {}

    def make_param_grid(self):
        return list(grid_search.ParameterGrid(self.cv_params))

    def make_splits_regression(self,y):
        return cv.ShuffleSplit(len(y),n_iter=10,test_size=0.2);

    def make_splits_classification(self,y):
        return cv.StratifiedShuffleSplit(y,n_iter=10,test_size=0.2);