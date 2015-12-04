__author__ = 'Aubrey'

from sklearn.neighbors import DistanceMetric
from sklearn.metrics import pairwise
import numpy as np
from numpy.linalg import norm
class CosineDistanceMetric(DistanceMetric):
    def __init__(self):
        pass

    def pairwise(self,X,Y):
        W = pairwise.pairwise_distances(X,Y,'cosine')

    def get_metric(self):
        return 'cosine'

    def __call__(self,X,Y, **kwargs):
        return 1 - X.dot(Y)/(norm(X)*norm(Y))