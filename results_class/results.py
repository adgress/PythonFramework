__author__ = 'Aubrey'

import numpy as np
import collections
import math
from data import data as data_lib
from loss_functions import loss_function as loss_function_lib
from utility import array_functions

aggregated_results = collections.namedtuple('AggregatedResults',['mean','low','high'])
processed_results = collections.namedtuple('ProcessedResults',['means','lows','highs'])

class ResultsContainer(object):
    def __init__(self, n=0):
        self.results_list = [None]*n

    def append(self,new_results):
        self.results_list.append(new_results)

    def set(self, new_results, ind, ind_sub=None):
        if ind_sub is None:
            self.results_list[ind] = new_results
        else:
            self.results_list[ind].results_list[ind_sub] = new_results

    @property
    def is_full(self):
        return all([x is not None for x in self.results_list])

class MethodResults(ResultsContainer):
    def __init__(self, n_exp, n_splits):
        super(MethodResults, self).__init__()
        self.results_list = list([ExperimentResults(n_splits) for i in range(n_exp)])
        pass

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
    def __init__(self, n):
        super(ExperimentResults, self).__init__(n)
        self.configs = []
        self.num_labels = None
        self.is_regression = True

    def aggregate_error(self, loss_function):
        errors = self.compute_error(loss_function)
        mean = errors.mean()
        #mean = np.percentile(errors,50)
        n = len(errors)
        zn = 1.96
        #if self.is_regression or mean > 1:
        if mean > 1:
            std = errors.std()
            se = std / math.sqrt(n)
            low = se*zn
            high = se*zn
        else:
            low = zn*math.sqrt(mean*(1-mean)/n)
            high = low
            #low = mean - np.percentile(errors,10)
            #high = np.percentile(errors,90) - mean

        agg_res = aggregated_results(mean,low,high)
        return agg_res

    def compute_error(self, loss_function):
        errors = np.empty(len(self.results_list))
        for i, f in enumerate(self.results_list):
            output = f.prediction
            #TODO: Check if we should use y or fu
            if output.fu.ndim > 1 and isinstance(loss_function, loss_function_lib.LogLoss):
                fu = output.fu[~output.is_train,:]
                true_fu = array_functions.make_label_matrix(output.true_y[~output.is_train]).toarray()
                errors[i] = loss_function.compute_score(fu,true_fu)
            else:
                errors[i] = loss_function.compute_score(output.y,output.true_y,~output.is_train)

        assert all(~np.isnan(errors))
        return errors

class FoldResults(object):
    def __init__(self):
        self.prediction = Output()
        self.estimated_error = None

    def compute_error(self,loss_function):
        return loss_function.compute_score(
            self.prediction.y,
            self.prediction.true_y,
            ~self.prediction.is_train
        )

class Output(data_lib.LabeledVector):
    def __init__(self,data=None,y=None):
        super(data_lib.LabeledVector, self).__init__()
        if data is not None:
            self.y = data.y
            self.is_train = data.is_train
            self.true_y = data.true_y
            self.type = data.type
        else:
            self.y = np.empty(0)
            self.is_train = np.empty(0)
            self.true_y = np.empty(0)
            self.type = np.empty(0)
        self.fu = np.empty(0)
        if y is not None:
            self.y = y
            self.fu = y

    def compute_error_train(self,loss_function):
        return loss_function.compute_score(
            self.y,
            self.true_y,
            self.is_train
        )

    def compute_error(self,loss_function):
        return loss_function.compute_score(
            self.y,
            self.true_y,
            ~self.is_train
        )

    def assert_input(self):
        assert not array_functions.has_invalid(self.fu)
        assert not array_functions.has_invalid(self.y)
        assert not array_functions.has_invalid(self.true_y)

class ClassificationOutput(Output):
    def __init__(self,data=None):
        super(ClassificationOutput,self).__init__(data)

    @property
    def y_discrete(self):
        return self.y.round()