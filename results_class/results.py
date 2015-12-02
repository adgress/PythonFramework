__author__ = 'Aubrey'

import numpy as np
import collections
import math
from loss_functions import loss_function as loss_function_lib

aggregated_results = collections.namedtuple('AggregatedResults',['mean','low','high'])
processed_results = collections.namedtuple('ProcessedResults',['means','lows','highs'])

class ResultsContainer(object):
    def __init__(self):
        self.results_list = []

    def append(self,new_results):
        self.results_list.append(new_results)

class MethodResults(ResultsContainer):
    def __init__(self):
        super(MethodResults, self).__init__()

    def compute_error_processed(self, loss_function):
        errors = self.compute_error(loss_function)
        means = [x.mean for x in errors]
        lows = [x.low for x in errors]
        highs = [x.high for x in errors]
        return processed_results(means,lows,highs)

    def compute_error(self, loss_function):
        errors = []
        for i, f in enumerate(self.results_list):
            errors.append(f.aggregate_error(loss_function))
        return errors

    @property
    def sizes(self):
        a = [x.num_labels for x in self.results_list]
        return np.asarray(a)

    def get_field(self, field):
        a = [getattr(x,field) for x in self.results_list]
        return np.asarray(a)



class ExperimentResults(ResultsContainer):
    def __init__(self):
        super(ExperimentResults, self).__init__()
        self.configs = []
        self.num_labels = None
        self.is_regression = True

    def aggregate_error(self, loss_function):
        errors = self.compute_error(loss_function)
        mean = errors.mean()
        n = len(errors)
        zn = 1.96
        if self.is_regression:
            std = errors.std()
            se = std / math.sqrt(n)
            low = se*zn
            high = se*zn
        else:
            low = zn*math.sqrt(mean*(1-mean)/n)

        agg_res = aggregated_results(mean,low,high)
        return agg_res

    def compute_error(self, loss_function):
        errors = np.empty(len(self.results_list))
        for i, f in enumerate(self.results_list):
            output = f.prediction
            errors[i] = loss_function.compute_score(output.y,output.true_y,~output.is_train)
        assert all(~np.isnan(errors))
        return errors

class FoldResults(object):
    def __init__(self):
        self.prediction = Output()
        self.actual = Output()

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