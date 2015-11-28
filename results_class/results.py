__author__ = 'Aubrey'

import numpy as np
import collections

class ExperimentResults(object):
    def __init__(self):
        pass

class FoldResults(object):
    def __init__(self):
        self.prediction = Output()
        self.actual = Output()

class ResultsVector(object):
    def __init__(self):
        self.x = None

class Output(object):
    def __init__(self,data=None):
        if data is not None:
            self.y = data.y
            self.is_train = data.is_train
            self.true_y = data.true_y
        else:
            self.y = np.empty(0)
            self.is_train = np.empty(0)
            self.type = np.empty(0)
        self.fu = np.empty(0)


class ClassificationOutput(Output):
    def __init__(self,data=None):
        super(ClassificationOutput,self).__init__(data)

    @property
    def y_discrete(self):
        return self.y.round()