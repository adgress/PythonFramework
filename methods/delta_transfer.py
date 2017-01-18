import scipy
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
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
        self.use_radius = None
        self.C3 = None
        self.use_l2 = True
        self.constant_b = getattr(configs, 'constant_b', False)
        self.linear_b = getattr(configs, 'linear_b', False)
        self.clip_b = getattr(configs, 'clip_b', False)
        self.sigma = None
        #self.transform = StandardScaler()
        self.transform = MinMaxScaler()

    def train(self, data):
        y_s = np.squeeze(data.y_s[:,0])
        y_t = np.squeeze(data.y_t[:,0])
        y = data.y
        if self.constant_b:
            self.g = (y_t - y_s).mean()
            return

        is_labeled = data.is_labeled
        labeled_inds = is_labeled.nonzero()[0]
        n_labeled = len(labeled_inds)
        if self.linear_b:
            g = cvx.Variable(data.p)
            b = cvx.Variable(1)
            x = self.transform.fit_transform(data.x[is_labeled,:])
            err = self.C3*y_t + (1 - self.C3)*(y_s + x*g + b) - y
            reg = cvx.square(cvx.norm2(g))
        else:
            g = cvx.Variable(n_labeled)
            if self.use_radius:
                W = array_functions.make_graph_radius(data.x[is_labeled,:], self.radius, self.configs.metric)
            else:
                #W = array_functions.make_graph_adjacent(data.x[is_labeled,:], self.configs.metric)
                #W = array_functions.make_graph_adjacent(data.x[is_labeled, :], self.configs.metric)
                W = array_functions.make_rbf(data.x[is_labeled, :], self.sigma, self.configs.metric)
            W = array_functions.try_toarray(W)
            W = .5*(W + W.T)
            if W.sum() > 0:
                W = W / W.sum()
            reg = 0
            if W.any():
                if self.use_fused_lasso:
                    reg = cvx_functions.create_fused_lasso(W, g)
                else:
                    L = array_functions.make_laplacian_with_W(W)
                    reg = cvx.quad_form(g,L)
                    #reg = g.T * L * g
            err = self.C3*y_t + (1 - self.C3)*(y_s+g) - y
        #err = y_s + g - y
        err_abs = cvx.abs(err)
        err_l2 = cvx.power(err,2)
        err_huber = cvx.huber(err, 2)
        if self.use_l2:
            loss = cvx.sum_entries(err_l2)
        else:
            loss = cvx.sum_entries(err_huber)
        #constraints = [g >= -2, g <= 2]
        #constraints = [g >= -4, g <= 0]
        #constraints = [g >= 4, g <= 4]
        if self.linear_b:
            constraints = [f(g,b,x) for f in self.configs.constraints]
        else:
            constraints = [f(g) for f in self.configs.constraints]
        obj = cvx.Minimize(loss + self.C * reg)
        #obj = cvx.Minimize(loss + self.C*reg + self.C2*cvx.norm(g))
        prob = cvx.Problem(obj,constraints)

        assert prob.is_dcp()
        try:
            prob.solve()
            if self.linear_b:
                b_value = b.value
                g_value = np.reshape(np.asarray(g.value),data.p)
            else:
                g_value = np.reshape(np.asarray(g.value),n_labeled)
        except:
            k = 0
            #assert prob.status is None
            print 'CVX problem: setting g = ' + str(k)
            g_value = k*np.ones(n_labeled)
            if self.linear_b:
                g_value = k*np.ones(data.p)
                b_value = 0
            print '\tC=' + str(self.C)
            print '\tC2=' + str(self.C2)
            print '\tC3=' + str(self.C3)
        if self.linear_b:
            self.g = g_value
            self.b = b_value
            g_pred = x.dot(g_value)
            self.g_min = g_pred.min()
            self.g_max = g_pred.max()
            return
        #labeled_train_data = data.get_subset(labeled_inds)
        training_data = data.get_subset(data.is_train)
        assert training_data.y.shape == g_value.shape
        training_data.is_regression = True
        training_data.y = g_value
        training_data.true_y = g_value

        self.g_nw.train_and_test(training_data)

    def combine_predictions(self,x,y_source,y_target,data_set_ids):
        data = data_lib.Data()
        data.x = x
        data.is_regression = True
        if self.constant_b:
            g = self.g
        elif self.linear_b:
            x = self.transform.transform(data.x)
            g = x.dot(self.g)
            if self.clip_b:
                g = array_functions.clip(g,self.g_min,self.g_max)
            g = g + self.b
        else:
            g = self.g_nw.predict(data).fu
        fu = self.C3*y_target + (1-self.C3)*(y_source + g)
        #fu = y_source + g
        return fu


    def predict_g(self, x, data_set_ids):
        if self.constant_b:
            g = self.g
        elif self.linear_b:
            x = self.transform.transform(x)
            g = x.dot(self.g)
            if self.clip_b:
                g = array_functions.clip(g,self.g_min,self.g_max)
            g = g + self.b
        else:
            g = super(CombinePredictionsDelta, self).predict_g(x)
        return g

    @property
    def prefix(self):
        s = 'DelTra'
        return s

class CombinePredictionsDeltaMultitask(CombinePredictionsDelta):
    def __init__(self, configs=None):
        super(CombinePredictionsDeltaMultitask, self).__init__(configs)
        self.use_radius = None
        self.C3 = None
        self.use_l2 = True
        self.constant_b = getattr(configs, 'constant_b', False)
        self.linear_b = getattr(configs, 'linear_b', False)
        self.clip_b = getattr(configs, 'clip_b', False)
        self.sigma = None
        #self.transform = StandardScaler()
        self.transform = MinMaxScaler()

    def train(self, data):
        y_s = np.squeeze(data.y_s[:, 0])
        y_t = np.squeeze(data.y_t[:, 0])
        ids = data.data_set_ids
        y = data.y

        assert self.linear_b


        is_labeled = data.is_labeled
        labeled_inds = is_labeled.nonzero()[0]
        labeled_ids = data.data_set_ids[labeled_inds]
        unique_ids = np.unique(labeled_ids)
        n_labeled = len(labeled_ids)
        if self.linear_b:
            g = [cvx.Variable(data.p) for i in range(unique_ids.size)]
            b = [cvx.Variable(1) for i in range(unique_ids.size)]
            reg = 0
            err = 0
            x = self.transform.fit_transform(data.x[labeled_inds, :])
            for i, id in enumerate(unique_ids):
                I = labeled_ids == id
                #x = self.transform.fit_transform(data.x[labeled_inds[I], :])
                err += cvx.sum_squares(self.C3 * y_t[I] + (1 - self.C3) * (y_s[I] + x[I, :] * g[i] + b[i]) - y[I])
                #reg += cvx.square(cvx.norm2(g[i]))
                reg += cvx.sum_squares(g[i])

        loss = err
        #currently only using l2 error
        '''
        err_l2 = cvx.power(err, 2)
        err_huber = cvx.huber(err, 2)
        if self.use_l2:
            loss = cvx.sum_entries(err_l2)
        else:
            loss = cvx.sum_entries(err_huber)
        '''
        # constraints = [g >= -2, g <= 2]
        # constraints = [g >= -4, g <= 0]
        # constraints = [g >= 4, g <= 4]
        if self.linear_b:
            constraints = [f(g, b, x) for f in self.configs.constraints]
        else:
            constraints = [f(g) for f in self.configs.constraints]
        assert unique_ids.size == 2
        reg_multitask = cvx.sum_squares(g[0] - g[1])
        #obj = cvx.Minimize(loss + self.C * reg + self.C2 * cvx.norm(g))
        obj = cvx.Minimize(loss + self.C * reg + self.reg_MT * reg_multitask)
        prob = cvx.Problem(obj, constraints)

        assert prob.is_dcp()
        try:
            #If C3 == 1, then just set 'g' and 'b' to 0
            assert self.C3 < 1
            val = prob.solve()
            if val == np.inf:
                print 'Inf'
                assert False
            if self.linear_b:
                b_value = [bi.value for bi in b]
                g_value = [np.reshape(np.asarray(gi.value), data.p) for gi in g]
            else:
                g_value = [np.reshape(np.asarray(gi.value), n_labeled) for gi in g]
            pass
        except:
            k = 0
            # assert prob.status is None
            if self.C3 != 1:
                print 'CVX problem: setting g = ' + str(k)
                print '\tC3=' + str(self.C3)
                print '\treg_MT=' + str(self.reg_MT)
            g_value = [k * np.ones(n_labeled) for i in range(unique_ids.size)]
            if self.linear_b:
                g_value = [k * np.ones(data.p) for i in range(unique_ids.size)]
                b_value = [0]*unique_ids.size
        if self.linear_b:
            self.g = g_value
            self.b = b_value
            self.g_min = np.inf
            self.g_max = -np.inf
            for i, id in enumerate(unique_ids):
                I = labeled_ids == id
                g_pred = x[I, :].dot(g_value[i]) + b_value[i]
                self.g_min = min(g_pred.min(), self.g_min)
                self.g_max = max(g_pred.max(), self.g_max)
            return

    def combine_predictions(self,x,y_source,y_target, data_set_ids):
        assert x.shape[0] == data_set_ids.shape[0]
        data = data_lib.Data()
        data.x = x
        data.is_regression = True
        if self.linear_b:
            unique_ids = np.unique(data_set_ids)
            x = self.transform.transform(data.x)
            g = np.zeros(x.shape[0])
            for i, id in enumerate(unique_ids):
                I = data_set_ids == id
                g[I] = x[I].dot(self.g[i])
                g[I] = g[I] + self.b[i]
            if self.clip_b:
                g = array_functions.clip(g, self.g_min, self.g_max)
        fu = self.C3*y_target + (1-self.C3)*(y_source + g)
        #fu = y_source + g
        return fu

    def predict_g(self, x, data_set_ids):
        assert x.shape[0] == data_set_ids.shape[0]
        if self.linear_b:
            unique_ids = np.unique(data_set_ids)
            x = self.transform.transform(x)
            g = np.zeros(x.shape[0])
            for i, id in enumerate(unique_ids):
                I = data_set_ids == id
                g[I] = x[I].dot(self.g[i])
                g[I] = g[I] + self.b[i]
            if self.clip_b:
                g = array_functions.clip(g,self.g_min,self.g_max)
        return g

    @property
    def prefix(self):
        s = 'DelTraMT'
        return s

class CombinePredictionsDeltaSMS(CombinePredictionsDelta):
    def __init__(self, configs=None):
        super(CombinePredictionsDeltaSMS, self).__init__(configs)
        self.g_nw = None
        self.include_scale = True

    def train(self, data):
        assert data.is_regression
        is_labeled = data.is_labeled
        y_s = data.y_s[is_labeled]
        y = data.y[is_labeled]
        assert not is_labeled.all()
        labeled_inds = is_labeled.nonzero()[0]
        n_labeled = len(labeled_inds)
        g = cvx.Variable(n_labeled)
        w = cvx.Variable(n_labeled)
        W_ll = array_functions.make_rbf(data.x[is_labeled,:], self.sigma, self.configs.metric)


        self.x = data.x[is_labeled,:]
        self.y = y

        self.R_ll = W_ll*np.linalg.inv(W_ll + self.C*np.eye(W_ll.shape[0]))
        R_ul = self.make_R_ul(data.x)
        err = y_s + self.R_ll*g - y
        err_l2 = cvx.power(err,2)
        reg = cvx.norm(R_ul*w - 1)
        loss = cvx.sum_entries(err_l2) + self.C2*reg
        constraints = []
        if not self.include_scale:
            constraints.append(w == 1)
        obj = cvx.Minimize(loss)
        prob = cvx.Problem(obj,constraints)

        assert prob.is_dcp()
        try:
            prob.solve()
            g_value = np.reshape(np.asarray(g.value),n_labeled)
            w_value = np.reshape(np.asarray(w.value),n_labeled)
        except:
            k = 0
            #assert prob.status is None
            print 'CVX problem: setting g = ' + str(k)
            print '\tC=' + str(self.C)
            print '\tC2=' + str(self.C2)
            print '\tsigma=' + str(self.sigma)
            g_value = k*np.ones(n_labeled)
            w_value = np.ones(n_labeled)
        self.g = g_value
        self.w = w_value

    def make_R_ul(self, x):
        W_ul = array_functions.make_rbf(x, self.sigma, self.configs.metric, self.x)
        R_ul = W_ul.dot(self.R_ll)
        return R_ul

    def predict_g(self, x):
        R_ul = self.make_R_ul(x)
        g = R_ul.dot(self.g)
        return g

    def predict_w(self, x):
        R_ul = self.make_R_ul(x)
        w = R_ul.dot(self.w)
        return w

    def combine_predictions(self,x,y_source,y_target):
        fu = np.multiply(self.predict_w(x), y_source) + self.predict_g(x)
        return fu

    @property
    def prefix(self):
        s = 'SMSTra'
        if getattr(self,'include_scale',False):
            s += '_scale'
        return s
