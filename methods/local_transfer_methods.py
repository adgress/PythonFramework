import copy

import cvxpy as cvx
import numpy as np

from loss_functions import loss_function
from methods import method, scipy_opt_methods
from methods.transfer_methods import FuseTransfer
from utility import array_functions
from utility import cvx_functions
from numpy import multiply
from numpy.linalg import norm

enable_plotting = False

class HypothesisTransfer(FuseTransfer):
    def __init__(self, configs=None):
        super(HypothesisTransfer, self).__init__(configs)
        self.cv_params = {}
        self.cv_params['a'] = np.asarray([0, .2, .4, .6, .8, 1],dtype='float64')
        self.target_learner = method.NadarayaWatsonMethod(configs)
        self.source_learner = method.NadarayaWatsonMethod(configs)
        self.base_learner = None

    def get_target_subset(self, data):
        if data.is_regression:
            target_data = data.get_transfer_subset(self.configs.target_labels, include_unlabeled=True)
        else:
            target_data = data.get_subset(data.is_target)
        return target_data

    def get_predictions(self, target_data):
        o = self.target_learner.predict_loo(target_data)
        o_source = self.source_learner.predict(target_data)
        is_labeled = target_data.is_labeled

        target_labels = self.configs.target_labels
        if self.use_estimated_f:
            o = self.target_learner.predict_loo(target_data.get_subset(is_labeled))
        if target_data.is_regression:
            y_t = array_functions.vec_to_2d(o.fu)
            y_s = array_functions.vec_to_2d(o_source.fu[is_labeled])
            y_true = array_functions.vec_to_2d(o.true_y)
        else:
            y_t = o.fu[:,target_labels]
            y_s = o_source.fu[:,target_labels]
            y_s = y_s[is_labeled,:]
            y_true = array_functions.make_label_matrix(o.true_y)[:,target_labels]
            y_true = array_functions.try_toarray(y_true)
        return (y_t, y_s, y_true)

    def train_and_test(self, data):
        #source_data = data.get_with_labels(self.configs.source_labels)
        if data.is_regression:
            source_data = data.get_transfer_subset(self.configs.source_labels.ravel(),include_unlabeled=True)
        else:
            source_data = data.get_transfer_subset(self.configs.source_labels.ravel(),include_unlabeled=False)
        source_data.set_target()
        source_data.set_train()
        source_data.reveal_labels(~source_data.is_labeled)
        if not data.is_regression:
            source_data.change_labels(self.configs.source_labels,self.configs.target_labels)
            source_data = source_data.rand_sample(.1)
        self.source_learner.train_and_test(source_data)
        target_data = data.get_transfer_subset(self.configs.target_labels, include_unlabeled=True)
        self.target_learner.train_and_test(target_data)
        return super(HypothesisTransfer, self).train_and_test(data)

    def train(self, data):
        pass

    def predict(self, data):
        o = self.target_learner.predict(data)
        is_target = data.is_target
        o_source = self.source_learner.predict(data.get_subset(is_target))
        if not data.is_regression:
            assert o.fu.ndim == 2
        else:
            assert o.fu.ndim == 1
            assert o_source.fu.ndim == 1
            o.fu = o.fu.reshape((o.fu.size,1))
            o_source.fu = o_source.fu.reshape((o_source.fu.size,1))
        for i in range(o.fu.shape[1]):
            fu_t = o.fu[is_target,i]
            fu_s = o_source.fu[:,i]
            o.fu[is_target,i] = self.a*fu_s + (1-self.a)*fu_t
            #o.fu[is_target] = np.multiply(o.fu[is_target],(1-self.g)) + np.multiply(self.g,o_source.fu)
        if data.is_regression:
            o.y = o.fu
        else:
            fu = array_functions.replace_invalid(o.fu,0,1)
            fu = array_functions.normalize_rows(fu)
            o.fu = fu
            o.y = fu.argmax(1)
        assert not (np.isnan(o.y)).any()
        assert not (np.isnan(o.fu)).any()
        return o

    @property
    def prefix(self):
        return 'HypothesisTransfer'

class LocalTransfer(HypothesisTransfer):
    def __init__(self, configs=None):
        super(LocalTransfer, self).__init__(configs)
        self.cv_params = {}
        #self.cv_params['sigma'] = 10**np.asarray(range(-4,4),dtype='float64')
        self.sigma = 100
        #self.cv_params['radius'] = np.asarray([.01, .05, .1, .15, .2],dtype='float64')
        self.radius = .05
        #self.cv_params['C'] = 10**np.asarray(range(-4,4),dtype='float64')
        self.cv_params['C'] = 10**np.asarray(range(-8,4),dtype='float64')
        self.cv_params['C'] = np.insert(self.cv_params['C'],0,0)
        #self.C = 1
        self.k = 3
        self.target_learner = method.NadarayaWatsonMethod(configs)
        self.source_learner = method.NadarayaWatsonMethod(configs)
        self.base_learner = None
        self.learn_g_just_labeled = True
        self.should_plot_g = False
        self.use_fused_lasso = True
        self.g_learner = None
        use_g_learner = False
        if use_g_learner:
            self.g_learner = scipy_opt_methods.ScipyOptCombinePrediction(configs)
            self.g_learner.max_value = 1
        #self.g_learner = None
        self.use_estimated_f = False
        self.metric = 'euclidean'

    def train_g_learner(self, target_data):
        y_t, y_s, y_true = self.get_predictions(target_data)

        is_labeled = target_data.is_labeled
        if target_data.is_regression:
            a = y_s - y_t
            b = y_t - y_true
        else:
            a = y_s[:,0] - y_t[:,0]
            b = y_t[:,0] - y_true[:,0]


        parametric_data = target_data.get_subset(is_labeled)
        parametric_data.a = a
        parametric_data.b = b
        parametric_data.set_defaults()
        #s = np.hstack((a,b))
        #s[parametric_data.x.argsort(0)]
        self.g_learner.C = self.C
        self.g_learner.cv_params = {}
        self.g_learner.train_and_test(parametric_data)

    def train_g_nonparametric_all(self, target_data):
        assert False, 'Do we still want to use this?  If so, then fix it'
        g = cvx.Variable(target_data.n)
        is_labeled = target_data.is_labeled
        labeled_inds = is_labeled.nonzero()[0]
        unlabeled_inds = (~is_labeled).nonzero()[0]
        sorted_inds = np.concatenate((labeled_inds,unlabeled_inds))
        n_labeled = len(labeled_inds)
        L = array_functions.make_laplacian_uniform(target_data.x,self.radius,metric) + .0001*np.identity(target_data.n)
        L = L[:,sorted_inds]
        L = L[sorted_inds,:]
        y_s = y_s[sorted_inds,:]
        reg = cvx.quad_form(g,L)

        loss = cvx.sum_entries(
            cvx.power(
                #y_s[:n_labeled,0] * g[:n_labeled] + y_t[:,0] * (1-g[:n_labeled]) - y_true[:,0],
                cvx.mul_elemwise(
                    y_s[:n_labeled,0],g[:n_labeled])
                        + cvx.mul_elemwise(y_t[:,0], (1-g[:n_labeled]))
                        - y_true[:,0],
                2
            )
        )
        constraints = [g >= 0, g <= .5]
        obj = cvx.Minimize(loss + self.C*reg)
        prob = cvx.Problem(obj,constraints)

        assert prob.is_dcp()
        try:
            prob.solve()
            inv_perm = array_functions.inv_permutation(sorted_inds)
            self.g = g.value[inv_perm]
            assert (self.g[sorted_inds] == g.value).all()
        except:
            #assert prob.status is None
            k = 0
            print 'CVX problem: setting g = ' + str(k)
            print '\tsigma=' + str(self.sigma)
            print '\tC=' + str(self.C)
            print '\tradius=' + str(self.radius)
            self.g = k*np.ones(target_data.n)
        if target_data.x.shape[1] == 1:
            self.g_x = target_data.x
            self.g_x_low = self.g[self.g_x[:,0] < .5]
            self.g_x_high = self.g[self.g_x[:,0] >= .5]

    def train_g_nonparametric(self, target_data):
        y_t, y_s, y_true = self.get_predictions(target_data)

        is_labeled = target_data.is_labeled
        labeled_inds = is_labeled.nonzero()[0]
        n_labeled = len(labeled_inds)
        g = cvx.Variable(n_labeled)
        '''
        L = array_functions.make_laplacian_uniform(target_data.x[labeled_inds,:],self.radius,metric) \
            + .0001*np.identity(n_labeled)
        '''
        L = array_functions.make_laplacian_kNN(target_data.x[labeled_inds,:],self.k,self.metric) \
            + .0001*np.identity(n_labeled)
        if self.use_fused_lasso:
            reg = cvx_functions.create_fused_lasso(-L, g)
        else:
            reg = cvx.quad_form(g,L)
        loss = cvx.sum_entries(
            cvx.power(
                cvx.mul_elemwise(y_s[:,0], g) + cvx.mul_elemwise(y_t[:,0], (1-g)) - y_true[:,0],
                2
            )
        )
        self.C = 0
        constraints = [g >= 0, g <= .5]
        obj = cvx.Minimize(loss + self.C*reg)
        prob = cvx.Problem(obj,constraints)

        assert prob.is_dcp()
        try:
            prob.solve()
            g_value = np.reshape(np.asarray(g.value),n_labeled)
        except:
            k = 0
            #assert prob.status is None
            print 'CVX problem: setting g = ' + str(k)
            print '\tsigma=' + str(self.sigma)
            print '\tC=' + str(self.C)
            print '\tradius=' + str(self.radius)
            g_value = k*np.ones(n_labeled)
        if self.should_plot_g and enable_plotting and target_data.x.shape[1] == 1:
            array_functions.plot_2d(target_data.x[labeled_inds,:],g_value)

        labeled_train_data = target_data.get_subset(labeled_inds)
        assert labeled_train_data.y.shape == g_value.shape
        g_nw = method.NadarayaWatsonMethod(copy.deepcopy(self.configs))
        labeled_train_data.is_regression = True
        labeled_train_data.y = g_value
        labeled_train_data.true_y = g_value
        g_nw.configs.loss_function = loss_function.MeanSquaredError()
        g_nw.tune_loo(labeled_train_data)
        g_nw.train(labeled_train_data)
        target_data.is_regression = True
        self.g = g_nw.predict(target_data).fu
        self.g[labeled_inds] = g_value
        assert not np.any(np.isnan(self.g))

    def train(self, data):
        target_data = self.get_target_subset(data)
        if self.g_learner is not None:
            self.train_g_learner(target_data)
        elif self.learn_g_just_labeled:
            self.train_g_nonparametric(target_data)
        else:
            self.train_g_nonparametric_all(target_data)

        if self.should_plot_g and enable_plotting and target_data.x.shape[1] == 1:
            x = np.linspace(0,1)
            g = self.g_learner.predict_g(x)
            array_functions.plot_2d(x,g)
            pass


    def predict(self, data):
        o = self.target_learner.predict(data)
        is_target = data.is_target
        o_source = self.source_learner.predict(data.get_subset(is_target))
        if not data.is_regression:
            assert o.fu.ndim == 2
        else:
            assert o.fu.ndim == 1
            assert o_source.fu.ndim == 1
            o.fu = o.fu.reshape((o.fu.size,1))
            o_source.fu = o_source.fu.reshape((o_source.fu.size,1))
        for i in range(o.fu.shape[1]):
            fu_t = o.fu[is_target,i]
            fu_s = o_source.fu[:,i]
            if self.g_learner is not None:
                pred = self.g_learner.combine_predictions(data.x[is_target,:],fu_s,fu_t)
            else:
                pred = np.multiply(fu_t,1-self.g) + np.multiply(fu_s,self.g)
            o.fu[is_target,i] = pred
            #o.fu[is_target] = np.multiply(o.fu[is_target],(1-self.g)) + np.multiply(self.g,o_source.fu)
        if data.is_regression:
            o.y = o.fu
        else:
            fu = array_functions.replace_invalid(o.fu,0,1)
            fu = array_functions.normalize_rows(fu)
            o.fu = fu
            o.y = fu.argmax(1)
        assert not (np.isnan(o.y)).any()
        assert not (np.isnan(o.fu)).any()
        return o

    @property
    def prefix(self):
        s = 'LocalTransfer'
        if 'g_learner' in self.__dict__ and self.g_learner is not None:
            s += '-Parametric'
        if 'use_estimated_f' in self.__dict__ and self.use_estimated_f:
            s += '-est_f'
        return s