import scipy
from methods import method
from configs import base_configs
from utility import array_functions
import numpy as np
from numpy.linalg import norm
import math
from results_class import results
from data import data as data_lib
from loss_functions import loss_function
import cvxpy as cvx
import copy
from utility import cvx_functions
from methods import scipy_opt_methods

class CombinePredictionsDelta(scipy_opt_methods.ScipyOptNonparametricHypothesisTransfer):
    def __init__(self, configs=None):
        super(CombinePredictionsDelta, self).__init__(configs)

    def train(self, data):
        y_s = np.squeeze(data.y_s[:,0])
        y_t = np.squeeze(data.y_t[:,0])
        y = data.y

        is_labeled = data.is_labeled
        labeled_inds = is_labeled.nonzero()[0]
        n_labeled = len(labeled_inds)
        g = cvx.Variable(n_labeled)
        W = array_functions.make_graph_adjacent(data.x[is_labeled,:], self.configs.metric)
        if self.use_fused_lasso:
            reg = cvx_functions.create_fused_lasso(W, g)
        else:
            assert False, 'Make Laplacian!'
            #reg = cvx.quad_form(g,L)
        err = self.C3*y_s + (1-self.C3)*(y_t + g) - y
        err_abs = cvx.abs(err)
        err_l2 = cvx.power(err,2)
        loss = cvx.sum_entries(err_abs)
        constraints = [g >= -2, g <= 2]
        obj = cvx.Minimize(loss + self.C*reg + self.C2*cvx.norm(g))
        prob = cvx.Problem(obj,constraints)

        assert prob.is_dcp()
        try:
            prob.solve()
            g_value = np.reshape(np.asarray(g.value),n_labeled)
        except:
            k = 0
            #assert prob.status is None
            print 'CVX problem: setting g = ' + str(k)
            print '\tC=' + str(self.C)
            print '\tC2=' + str(self.C2)
            g_value = k*np.ones(n_labeled)

        labeled_train_data = data.get_subset(labeled_inds)
        assert labeled_train_data.y.shape == g_value.shape
        labeled_train_data.is_regression = True
        labeled_train_data.y = g_value
        labeled_train_data.true_y = g_value

        self.g_nw.tune_loo(labeled_train_data)
        self.g_nw.train_and_test(labeled_train_data)

    def combine_predictions(self,x,y_source,y_target):
        data = data_lib.Data()
        data.x = x
        data.is_regression = True
        g = self.g_nw.predict(data).fu
        fu = self.C3*y_target + (1-self.C3)*(y_source + g)
        return fu

    @property
    def prefix(self):
        s = 'DelTra'
        return s