top__author__ = 'Aubrey'

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
        '''
        hyper_params = []
        for i, curr_results in enumerate(self.results_list):
            hps_C = np.asarray([r.C for r in curr_results.results_list])
            hps_C2 = np.asarray([r.C2 for r in curr_results.results_list])
            d = {
                'C': hps_C,
                'C2': hps_C2
            }
            hyper_params.append(d)
        mean_Cs = np.asarray([np.log10(d['C']).mean() for d in hyper_params])
        mean_C2s = np.asarray([np.log10(d['C2']).mean() for d in hyper_params])

        var_Cs = np.asarray([np.log10(d['C']).var() for d in hyper_params])
        var_C2s = np.asarray([np.log10(d['C2']).var() for d in hyper_params])
        '''
        means = [x.mean for x in errors]
        lows = [x.low for x in errors]
        highs = [x.high for x in errors]
        return processed_results(means,lows,highs)

    def compute_error(self, loss_function):
        errors = []
        for i, f in enumerate(self.results_list):
            e = f.aggregate_error(loss_function)
            if len(self.results_list) == 1 and len(e) > 1:
                errors = e
            else:
                errors.append(e[0])
        return errors

    @property
    def sizes(self):
        a = [x.num_labels for x in self.results_list]
        return np.asarray(a)

    def get_field(self, field):
        a = [getattr(x,field) for x in self.results_list]
        return np.asarray(a)

class ActiveMethodResults(MethodResults):
    def __init__(self, n_exp, n_splits):
        super(ActiveMethodResults, self).__init__(n_exp, n_splits)

    @property
    def sizes(self):
        a = [x.num_labels for x in self.results_list]
        num_iterations = len(self.results_list[0].results_list[0].results_list)
        return np.asarray(range(num_iterations))


class ExperimentResults(ResultsContainer):
    def __init__(self, n):
        super(ExperimentResults, self).__init__(n)
        self.configs = []
        self.num_labels = None
        self.is_regression = True

    def aggregate_error(self, loss_function):
        agg_res = []
        errors = self.compute_error(loss_function)
        for i in range(errors.shape[1]):
            #mean = errors.mean()

            e = errors[:,i]

            sorted = np.sort(e)
            #I = (e <= sorted[-3])
            I = (e <= sorted[-1])
            e = e[I]

            #mean = np.percentile(errors[:,i],50)

            mean = e.mean()
            n = errors.shape[0]
            zn = 1.96
            #if self.is_regression or mean > 1:
            if mean > 1 or True:
                #std = errors[:,i].std()
                std = e.std()
                se = std / math.sqrt(n)
                low = se*zn
                high = se*zn
            else:
                low = zn*math.sqrt(mean*(1-mean)/n)
                high = low
                #low = mean - np.percentile(errors,10)
                #high = np.percentile(errors,90) - mean

            agg_res.append(aggregated_results(mean,low,high))
        return agg_res

    def compute_error(self, loss_function):
        for i, f in enumerate(self.results_list):
            e = f.compute_error(loss_function)
            e = np.asarray(e)
            if i == 0:
                errors = np.empty((len(self.results_list),e.size))
            errors[i,:] = e
        assert np.all(~np.isnan(errors))
        return errors


class FoldResults(object):
    def __init__(self):
        self.prediction = Output()
        self.estimated_error = None

    def compute_error(self,loss_function):
        #TODO: Check if we should use y or fu
        if self.prediction.fu.ndim > 1 and isinstance(loss_function, loss_function_lib.LogLoss):
            assert False, 'Update this'
            #fu = self.prediction.fu[~self.prediction.is_train,:]
            #true_fu = array_functions.make_label_matrix(output.true_y[~self.prediction.is_train]).toarray()
            #return loss_function.compute_score(fu,true_fu)
        #return loss_function.compute_score(self.prediction.y,self.prediction.true_y,~self.prediction.is_train)
        return self.prediction.compute_error(loss_function)

class ActiveFoldResults(ResultsContainer):
    def __init__(self, num_iterations):
        super(ActiveFoldResults, self).__init__(num_iterations)

    def compute_error(self,loss_function):
        errors = np.empty(len(self.results_list))
        for i, f in enumerate(self.results_list):
            errors[i] = f.compute_error(loss_function)
        assert all(~np.isnan(errors))
        return errors

class ActiveIterationResults(object):
    def __init__(self, fold_results=None, queried_idx=None):
        self.fold_results = fold_results
        self.queried_idx = queried_idx

    def compute_error(self,loss_function):
        return self.fold_results.compute_error(loss_function)

class Output(data_lib.LabeledVector):
    def __init__(self,data=None,y=None):
        super(Output, self).__init__()
        if data is not None:
            self.y = data.y
            self.is_train = data.is_train
            self.true_y = data.true_y
            self.type = data.type
            self.fu = np.zeros(data.y.shape)
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
        '''
        return loss_function.compute_score(
            self.y,
            self.true_y,
            ~self.is_train
        )
        '''
        return loss_function.compute_score(self)

    def assert_input(self):
        assert not array_functions.has_invalid(self.fu)
        assert not array_functions.has_invalid(self.y)
        assert not array_functions.has_invalid(self.true_y)

class RelativeRegressionOutput(Output):
    def __init__(self,data=None,y=None,pairwise_results=None):
        super(RelativeRegressionOutput, self).__init__(data, y)
        self.is_pairwise_correct = pairwise_results
        self.is_train_pairwise = data.is_train_pairwise

    def compute_error_train(self,loss_function):
        loss = super(RelativeRegressionOutput, self).compute_error_train(loss_function)
        num_train = self.is_train.sum()
        num_pairwise = self.is_train_pairwise.sum()
        avg_loss = loss / num_train
        pairwise_loss = (~self.is_pairwise_correct[self.is_train_pairwise]).sum()*avg_loss / num_pairwise
        return avg_loss + pairwise_loss

    def compute_error(self,loss_function):
        loss = super(RelativeRegressionOutput, self).compute_error(loss_function)
        num_test = (~self.is_train).sum()
        num_pairwise = (~self.is_train_pairwise).sum()
        if num_pairwise == 0:
            return loss
        avg_loss = loss/num_test
        #pairwise_loss = (~self.is_pairwise_correct[~self.is_train_pairwise]).sum()*avg_loss / num_pairwise
        pairwise_loss = (~self.is_pairwise_correct[~self.is_train_pairwise]).sum()*avg_loss
        return avg_loss + pairwise_loss

class ClassificationOutput(Output):
    def __init__(self,data=None):
        super(ClassificationOutput,self).__init__(data)

    @property
    def y_discrete(self):
        return self.y.round()