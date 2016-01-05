import copy

import cvxpy as cvx
import numpy as np

from loss_functions import loss_function
from methods import method, scipy_opt_methods
from methods.transfer_methods import FuseTransfer
from methods.transfer_methods import TargetTranfer
from utility import array_functions
from utility import cvx_functions
from numpy import multiply
from numpy.linalg import norm
from data import data as data_lib
from utility import helper_functions

if helper_functions.is_laptop():
    enable_plotting = True
else:
    enable_plotting = False

class HypothesisTransfer(method.Method):
    def __init__(self, configs=None):
        super(HypothesisTransfer, self).__init__(configs)
        self.cv_params = {}
        self.cv_params['a'] = np.asarray([0, .2, .4, .6, .8, 1],dtype='float64')
        self.target_learner = method.NadarayaWatsonMethod(configs)
        self.source_learner = method.NadarayaWatsonMethod(configs)
        self.target_learner.quiet = True
        self.source_learner.quiet = True
        self.base_learner = None
        self.use_oracle = False

    def _prepare_data(self, data, include_unlabeled=True):
        target_labels = self.configs.target_labels
        data_copy = data.get_transfer_subset(target_labels,include_unlabeled=include_unlabeled)
        #data_copy = data.get_with_labels(target_labels)
        return data_copy

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
        if data.is_regression:
            source_data = data.get_transfer_subset(self.configs.source_labels.ravel(),include_unlabeled=False)
        else:
            source_data = data.get_transfer_subset(self.configs.source_labels.ravel(),include_unlabeled=False)
        source_data.set_target()
        source_data.set_train()
        source_data.reveal_labels(~source_data.is_labeled)
        if source_data.is_regression:
            source_data.data_set_ids[:] = self.configs.target_labels[0]
        if self.use_oracle:
            oracle_labels = self.configs.oracle_labels
            source_data = source_data.get_transfer_subset(oracle_labels.ravel(),include_unlabeled=False)
        if not data.is_regression:
            source_data.change_labels(self.configs.source_labels,self.configs.target_labels)
            source_data = source_data.rand_sample(.1)

        viz_mds = False
        if viz_mds:
            source_labels = self.configs.source_labels
            target_labels = self.configs.target_labels
            data.data_set_ids[:] = 0
            data.data_set_ids[array_functions.find_set(data.y,source_labels[0,:])] = 1
            data.data_set_ids[array_functions.find_set(data.y,source_labels[1,:])] = 2
            data.change_labels(source_labels,target_labels)
            array_functions.plot_MDS(data.x,data.true_y,data.data_set_ids)

        self.source_learner.train_and_test(source_data)

        data_copy = self._prepare_data(data,include_unlabeled=True)
        data_copy = data_copy.get_transfer_subset(self.configs.target_labels, include_unlabeled=True)
        data_copy = data_copy.get_subset(data_copy.is_target)
        return super(HypothesisTransfer, self).train_and_test(data_copy)

    def train(self, data):
        #pass
        target_data = data.get_transfer_subset(self.configs.target_labels, include_unlabeled=True)
        self.target_learner.train_and_test(target_data)

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
        fu_orig = o.fu
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
        s = 'HypothesisTransfer'
        if 'use_oracle' in self.__dict__ and self.use_oracle:
            s += '-Oracle'
        return s

class LocalTransfer(HypothesisTransfer):
    def __init__(self, configs=None):
        super(LocalTransfer, self).__init__(configs)
        self.cv_params = {}
        #self.cv_params['sigma'] = 10**np.asarray(range(-4,4),dtype='float64')
        self.sigma = 100
        #self.cv_params['radius'] = np.asarray([.01, .05, .1, .15, .2],dtype='float64')
        self.radius = .05
        #self.cv_params['C'] = 10**np.asarray(range(-4,4),dtype='float64')
        self.cv_params['C'] = 10**np.asarray(range(-6,6),dtype='float64')
        if self.configs.use_fused_lasso:
            self.cv_params['C'] = np.asarray([1,5,10,50,100,500,1000000])
        self.cv_params['C'] = np.insert(self.cv_params['C'],0,0)

        self.cv_params['C2'] = np.asarray([0,.001,.01,.1,1,10,100,1000])

        if not self.configs.use_reg2:
            self.cv_params['C2'] = np.asarray([0])

        #self.C = 1
        self.k = 3
        #self.cv_params['k'] = np.asarray([1,3,5,7])
        self.target_learner = method.NadarayaWatsonMethod(configs)
        self.source_learner = method.NadarayaWatsonMethod(configs)
        self.base_learner = None
        self.learn_g_just_labeled = True
        self.should_plot_g = False
        self.use_fused_lasso = configs.use_fused_lasso
        self.use_oracle = False
        self.g_learner = None
        use_g_learner = configs.use_g_learner

        if use_g_learner:
            #self.g_learner = scipy_opt_methods.ScipyOptCombinePrediction(configs)
            self.g_learner = scipy_opt_methods.ScipyOptNonparametricHypothesisTransfer(configs)
            self.max_value = .5
            self.g_learner.max_value = self.max_value
        self.no_reg = self.configs.no_reg
        if self.no_reg:
            self.cv_params['C'] = np.zeros(1)
        #self.g_learner = None
        self.use_estimated_f = False
        #self.metric = 'euclidean'
        self.metric = configs.metric
        self.target_learner.quiet = True
        self.source_learner.quiet = True
        if self.g_learner is not None:
            self.g_learner.quiet = True


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
        parametric_data.y_s = y_s
        parametric_data.y_t = y_t
        parametric_data.set_defaults()
        #s = np.hstack((a,b))
        #s[parametric_data.x.argsort(0)]
        self.g_learner.C = self.C
        self.g_learner.C2 = self.C2
        self.g_learner.cv_params = {}
        self.g_learner.train_and_test(parametric_data)
        '''
        I = parametric_data.x.argsort(0)
        g_value = self.g_learner.predict_g(parametric_data.x)
        x = parametric_data.x
        x = np.squeeze(x)
        a = np.hstack((g_value[I], x[I]))
        print str(a)
        print 'C:' + str(self.C)
        '''
        pass

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
        constraints = [g >= 0, g <= .5]
        #constraints += [g[0] == .5, g[-1] == 0]
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
        '''
        a =np.hstack((g_value[labeled_train_data.x.argsort(0)], np.sort(labeled_train_data.x,0)))
        print str(a)
        print 'g_nw sigma: ' + str(g_nw.sigma)
        print 'C:' + str(self.C)
        '''
        target_data.is_regression = True
        self.g = g_nw.predict(target_data).fu
        self.g[labeled_inds] = g_value
        assert not np.any(np.isnan(self.g))

    def train_and_test(self, data):
        target_labels = self.configs.target_labels
        target_data = data.get_transfer_subset(target_labels,include_unlabeled=True)
        self.target_learner.train_and_test(target_data)
        results =  super(LocalTransfer, self).train_and_test(data)
        #print self.g_learner.g
        return results

    def plot_g(self):
        x = np.linspace(0,1)
        x = array_functions.vec_to_2d(x)
        g_orig = self.g_learner.predict_g(x)
        g = 1 / (1+g_orig)
        array_functions.plot_2d(x,g)
        pass

    def plot_source(self):
        x = np.linspace(0,1)
        x = array_functions.vec_to_2d(x)
        d = data_lib.Data()
        d.x = x
        d.y = np.nan*np.ones(x.shape[0])
        d.is_regression = True
        o = self.source_learner.predict(d)
        array_functions.plot_2d(x, o.y)

    def plot_target(self):
        x = np.linspace(0,1)
        x = array_functions.vec_to_2d(x)
        d = data_lib.Data()
        d.x = x
        d.y = np.nan*np.ones(x.shape[0])
        d.is_regression = True
        o = self.target_learner.predict(d)
        array_functions.plot_2d(x, o.y)

    def train(self, data):
        target_data = self.get_target_subset(data)
        #self.target_learner.train_and_test(target_data)
        self.target_learner.train(target_data)
        if self.g_learner is not None:
            self.train_g_learner(target_data)
        elif self.learn_g_just_labeled:
            self.train_g_nonparametric(target_data)
        else:
            self.train_g_nonparametric_all(target_data)
        I = target_data.is_labeled
        plot_functions = False
        if plot_functions:
            self.plot_target()
            self.plot_source()
            self.plot_g()
        if self.should_plot_g and enable_plotting and target_data.x.shape[1] == 1:
            self.plot_g()
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
        if 'use_oracle' in self.__dict__ and self.use_oracle:
            s += '-Oracle'
        if 'no_reg' in self.__dict__ and self.no_reg:
            s += '-no_reg'
        if 'g_learner' in self.__dict__ and self.g_learner is not None:
            s += '-' + self.g_learner.prefix
        elif 'use_fused_lasso' in self.__dict__ and self.use_fused_lasso:
            s += '-l1'
        else:
            s += '-l2'
        if 'use_estimated_f' in self.__dict__ and self.use_estimated_f:
            s += '-est_f'
        if 'max_value' in self.__dict__ and self.max_value != 1:
            s += '-max_value=' + str(self.max_value)
        return s