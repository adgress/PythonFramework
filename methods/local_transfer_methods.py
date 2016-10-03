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
from scipy import optimize
from methods import delta_transfer
import scipy

if helper_functions.is_laptop():
    enable_plotting = False
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

    def get_source_data(self, data):
        if data.is_regression:
            source_data = data.get_transfer_subset(self.configs.source_labels.ravel(), include_unlabeled=False)
        else:
            source_data = data.get_transfer_subset(self.configs.source_labels.ravel(), include_unlabeled=False)
        source_data.set_target()
        source_data.set_train()
        source_data.reveal_labels(~source_data.is_labeled)
        if source_data.is_regression:
            source_data.data_set_ids[:] = self.configs.target_labels[0]
        if self.use_oracle:
            oracle_labels = self.configs.oracle_labels
            source_data = source_data.get_transfer_subset(oracle_labels.ravel(), include_unlabeled=False)
        if not data.is_regression:
            source_data.change_labels(self.configs.source_labels, self.configs.target_labels)
            source_data = source_data.rand_sample(.1)
        return source_data

    def train_and_test(self, data):
        source_data = self.get_source_data(data)

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
            self.cv_params['C'] = np.asarray([1,3,5,10,20])
            #self.cv_params['C'] = np.asarray([1,5,10,50,100,500,1000000])
        self.cv_params['C'] = np.insert(self.cv_params['C'],0,0)


        self.cv_params['C2'] = np.asarray([0,.001,.01,.1,1,10,100,1000])
        if not self.configs.use_reg2:
            self.cv_params['C2'] = np.asarray([0])
        self.cv_params['C2'] = np.zeros(1)
        self.k = 1
        #self.cv_params['k'] = np.asarray([1,2,4])
        #self.cv_params['radius'] = np.asarray([.05, .1, .2])

        self.target_learner = method.NadarayaWatsonMethod(configs)
        self.source_learner = method.NadarayaWatsonMethod(configs)
        self.base_learner = None
        self.learn_g_just_labeled = True
        self.should_plot_g = False
        self.use_fused_lasso = configs.use_fused_lasso
        self.use_oracle = False
        self.g_learner = None
        #self.g_supervised = True
        self.g_supervised = False
        use_g_learner = configs.use_g_learner
        self.include_bias = False

        if use_g_learner:
            #self.g_learner = scipy_opt_methods.ScipyOptCombinePrediction(configs)
            self.g_learner = scipy_opt_methods.ScipyOptNonparametricHypothesisTransfer(configs)
            self.g_learner.g_supervised = self.g_supervised
            self.max_value = .5
            self.g_learner.max_value = self.max_value
            self.g_learner.include_bias = self.include_bias
        self.no_reg = self.configs.no_reg
        if self.no_reg:
            self.cv_params['C'] = np.zeros(1)
            self.cv_params['C2'] = np.zeros(1)
            self.cv_params['radius'] = np.zeros(1)
        #self.g_learner = None
        self.use_estimated_f = False
        #self.metric = 'euclidean'
        self.metric = configs.metric
        self.target_learner.quiet = True
        self.source_learner.quiet = True
        if self.g_learner is not None:
            self.g_learner.quiet = True

        self.quiet = False


    def train_g_learner(self, target_data):
        target_data = target_data.get_subset(target_data.is_train)
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
        parametric_data.set_target()
        parametric_data.set_train()
        #assert target_data.is_regression
        if target_data.is_regression:
            parametric_data.data_set_ids[:] = self.configs.target_labels[0]
        #s = np.hstack((a,b))
        #s[parametric_data.x.argsort(0)]
        self.g_learner.C = self.C
        self.g_learner.C2 = self.C2
        self.g_learner.k = self.k
        self.g_learner.radius = self.radius
        self.g_learner.sigma = self.sigma
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
        assert target_data.n > 0
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
                if data.x.shape[1] == 1:
                    x = scipy.linspace(data.x.min(),data.x.max(),100)
                    x = array_functions.vec_to_2d(x)
                    g = self.g_learner.predict_g(x)
                    o.x = x
                    o.g = g
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
        if data.x.shape[1] == 1:
            x = array_functions.vec_to_2d(scipy.linspace(data.x.min(),data.x.max(),100))
            o.linspace_x = x
            o.linspace_g = self.g_learner.predict_g(x)

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
        '''
        if 'max_value' in self.__dict__ and self.max_value != 1:
            s += '-max_value=' + str(self.max_value)
        '''
        if 'g_supervised' in self.__dict__ and self.g_supervised:
            s += '-g_sup'
        if 'include_bias' in self.__dict__ and self.include_bias:
            s += '-bias'
        return s


class OffsetTransfer(HypothesisTransfer):
    def __init__(self, configs=None):
        super(OffsetTransfer, self).__init__(configs)
        self.cv_params = dict()
        self.g_learner = method.NadarayaWatsonMethod(configs)
        self.x_source = None
        self.y_source_new = None


    def train_and_test(self, data):
        source_data = self.get_source_data(data)
        self.source_learner.train_and_test(source_data)
        '''
        data_copy = self._prepare_data(data, include_unlabeled=True)
        data_copy = data_copy.get_transfer_subset(self.configs.target_labels, include_unlabeled=True)
        data_copy = data_copy.get_subset(data_copy.is_target)
        '''
        data_copy = copy.deepcopy(data)
        return super(HypothesisTransfer, self).train_and_test(data_copy)

    def train(self, data):
        target_data = self.get_target_subset(data)
        target_data = target_data.get_subset(target_data.is_labeled)
        self.g_learner.train_and_test(target_data)
        source_data = self.get_source_data(data)
        x_source = source_data.x
        y_source_new = source_data.y + self.g_learner.predict(source_data).y
        source_data_new = data_lib.Data(x_source, y_source_new)

        all_data = copy.deepcopy(target_data)
        all_data.combine(source_data_new)
        self.target_learner.train_and_test(all_data)

    def predict(self, data):
        o = self.target_learner.predict(data)
        return o

    @property
    def prefix(self):
        s = 'OffsetTransfer'
        return s


class LocalTransferDelta(LocalTransfer):
    def __init__(self, configs=None):
        super(LocalTransferDelta, self).__init__(configs)
        self.C = None
        self.C2 = 0
        self.C3 = None
        self.radius = None
        self.cv_params = {}
        self.cv_params['radius'] = np.asarray([.05, .1, .2],dtype='float64')
        vals = [0] + list(range(-6,6))
        vals.reverse()
        self.cv_params['C'] = 10**np.asarray(vals,dtype='float64')
        self.cv_params['C3'] = np.asarray([0, .2, .4, .6, .8, 1])
        self.target_learner = method.NadarayaWatsonMethod(configs)
        self.source_learner = method.NadarayaWatsonMethod(configs)

        self.use_l2 = True

        self.g_learner = delta_transfer.CombinePredictionsDelta(configs)
        self.g_learner.quiet = True
        self.g_learner.use_l2 = self.use_l2
        self.g_learner.use_fused_lasso = configs.use_fused_lasso

        self.metric = configs.metric
        self.quiet = False
        self.no_C3 = configs.no_C3
        self.constant_b = configs.constant_b
        self.use_radius = configs.use_radius
        self.linear_b = configs.linear_b
        self.clip_b = configs.clip_b
        if self.constant_b:
            del self.cv_params['radius']
            del self.cv_params['C']
        if self.linear_b:
            del self.cv_params['radius']
        if not self.use_radius:
            del self.cv_params['radius']
        if self.no_C3:
            del self.cv_params['C3']
            self.C3 = 0

    def train_g_learner(self, target_data):
        self.g_learner.C3 = self.C3
        self.g_learner.use_radius = self.use_radius
        super(LocalTransferDelta, self).train_g_learner(target_data)
        pass

    def train_and_test(self, data):
        r = super(LocalTransferDelta, self).train_and_test(data)
        '''
        self.plot_g()
        I = np.squeeze(self.g_learner.g_nw.x.argsort(0))
        sorted_x = self.g_learner.g_nw.x[I,:]
        sorted_g = self.g_learner.g_nw.y[I]
        print sorted_x.T
        print sorted_g
        print self.g_learner.g_nw.sigma
        '''
        return r

    def plot_g(self):
        x = np.linspace(0,1)
        x = array_functions.vec_to_2d(x)
        g = self.g_learner.predict_g(x)
        array_functions.plot_2d(x,g)
        pass

    @property
    def prefix(self):
        s = 'LocalTransferDelta'
        is_nonparametric = not (getattr(self,'linear_b',False) or getattr(self,'constant_b',False))
        if getattr(self, 'no_C3', False):
            s += '_C3=0'
        if getattr(self, 'use_radius', False):
            s += '_radius'
        if getattr(self.configs, 'constraints', []):
            s += '_cons'
        if getattr(self, 'use_l2', False):
            s += '_l2'
        if getattr(self, 'constant_b', False):
            s += '_constant-b'
        if getattr(self, 'linear_b', False):
            s += '_linear-b'
            if getattr(self, 'clip_b', False):
                s += '_clip-b'
        if getattr(self.configs, 'use_validation', False):
            s += '_use-val'
        if not self.use_fused_lasso and is_nonparametric:
            s += '_lap-reg'
        return s


class LocalTransferDeltaSMS(LocalTransferDelta):
    def __init__(self, configs=None):
        super(LocalTransferDeltaSMS, self).__init__(configs)
        self.C2 = 0
        self.include_scale = configs.include_scale
        self.cv_params = {}
        vals = list(range(-4,5))
        vals.reverse()
        #vals = [0,1,2]
        self.cv_params['sigma'] = 10**np.asarray(vals,dtype='float64')
        self.cv_params['C'] = 10**np.asarray(vals,dtype='float64')
        if self.include_scale:
            self.cv_params['C2'] = 10**np.asarray(vals,dtype='float64')
        self.g_learner = delta_transfer.CombinePredictionsDeltaSMS(configs)
        self.g_learner.quiet = True
        self.quiet = False
        self.num_splits = 5


    def train_g_learner(self, target_data):
        target_data = copy.deepcopy(target_data)
        target_data.remove_test_labels()
        assert (~target_data.is_labeled[~target_data.is_train]).all()
        o_source = self.source_learner.predict(target_data)
        target_data.y_s = o_source.fu
        target_data.y_t = None
        is_labeled = target_data.is_labeled
        self.g_learner.C = self.C
        self.g_learner.C2 = self.C2
        self.g_learner.sigma = self.sigma
        self.g_learner.cv_params = {}
        self.g_learner.train_and_test(target_data)
        pass

    @property
    def prefix(self):
        s = 'LocalTransferDeltaSMS'
        if getattr(self, 'include_scale', False):
            s += '_scale'
        return s

class IWTLTransfer(method.Method):
    def __init__(self, configs=None):
        super(IWTLTransfer, self).__init__(configs)
        self.cv_params = {}
        self.cv_params['sigma'] = 10**np.asarray(range(-4,4),dtype='float64')
        self.cv_params['C'] = 10**np.asarray(range(-4,4),dtype='float64')
        self.target_learner = method.NadarayaWatsonMethod(configs)
        self.source_learner = method.NadarayaWatsonMethod(configs)
        self.target_learner.quiet = True
        self.source_learner.quiet = True
        self.base_learner = None
        self.metric = self.configs.metric

    def _prepare_data(self, data, include_unlabeled=True):
        target_labels = self.configs.target_labels
        data_copy = data.get_transfer_subset(target_labels,include_unlabeled=include_unlabeled)
        #data_copy = data.get_with_labels(target_labels)
        return data_copy

    def get_predictions(self, target_data):
        '''
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
        '''
        assert target_data.is_regression
        o = self.source_learner.predict(target_data)
        is_labeled = target_data.is_labeled
        y_s = array_functions.vec_to_2d(o.fu[is_labeled])
        y_true = array_functions.vec_to_2d(o.true_y[is_labeled])
        return (y_s, y_true)

    def train_and_test(self, data):
        assert data.is_regression
        if data.is_regression:
            source_data = data.get_transfer_subset(self.configs.source_labels.ravel(),include_unlabeled=False)
        else:
            source_data = data.get_transfer_subset(self.configs.source_labels.ravel(),include_unlabeled=False)
        source_data.set_target()
        source_data.set_train()
        source_data.reveal_labels(~source_data.is_labeled)
        if source_data.is_regression:
            source_data.data_set_ids[:] = self.configs.target_labels[0]
        if not data.is_regression:
            source_data.change_labels(self.configs.source_labels,self.configs.target_labels)
            source_data = source_data.rand_sample(.1)

        self.source_learner.train_and_test(source_data)

        data_copy = data.get_transfer_subset(self.configs.labels_to_keep)
        is_source = array_functions.find_set(data_copy.data_set_ids,self.configs.source_labels)
        data_copy.type[is_source] = data_lib.TYPE_SOURCE
        assert data_copy.is_source.any()
        return super(IWTLTransfer, self).train_and_test(data_copy)

    def train(self, data):
        assert data.is_regression
        y_s, y_true = self.get_predictions(data)
        I = data.is_target & data.is_labeled
        #y_s = y_s[I]
        y_s = data.y[data.is_source]
        y_true = data.true_y[I]

        x_s = data.x[data.is_source]
        x_s = array_functions.append_column(x_s, data.y[data.is_source])
        x_s = array_functions.standardize(x_s)
        x_t = data.x[I]
        x_t = array_functions.append_column(x_t, data.y[I])
        x_t = array_functions.standardize(x_t)
        Wrbf = array_functions.make_rbf(x_t, self.sigma, self.metric, x2=x_s)
        S = array_functions.make_smoothing_matrix(Wrbf)
        w = cvx.Variable(x_s.shape[0])
        constraints = [w >= 0]
        reg = cvx.norm(w)**2
        loss = cvx.sum_entries(
            cvx.power(
                S*cvx.diag(w)*y_s - y_true,2
            )
        )
        obj = cvx.Minimize(loss + self.C*reg)
        prob = cvx.Problem(obj,constraints)
        assert prob.is_dcp()
        try:
            prob.solve()
            #g_value = np.reshape(np.asarray(g.value),n_labeled)
            w_value = w.value
        except:
            k = 0
            #assert prob.status is None
            print 'CVX problem: setting g = ' + str(k)
            print '\tsigma=' + str(self.sigma)
            print '\tC=' + str(self.C)
            w_value = k*np.ones(x_s.shape[0])

        all_data = data.get_transfer_subset(self.configs.labels_to_keep,include_unlabeled=True)
        all_data.instance_weights = np.ones(all_data.n)
        all_data.instance_weights[all_data.is_source] = w.value
        self.instance_weights = all_data.instance_weights
        self.target_learner.train_and_test(all_data)

        self.x = all_data.x[all_data.is_source]
        self.w = all_data.instance_weights[all_data.is_source]

    def predict(self, data):
        o = self.target_learner.predict(data)
        o.x = self.x
        o.w = self.w
        return o

    @property
    def prefix(self):
        s = 'IWTL'
        return s


class SMSTransfer(method.Method):
    def __init__(self, configs=None):
        super(SMSTransfer, self).__init__(configs)
        self.cv_params = {}

        self.cv_params['sigma'] = 10**np.asarray(range(-4,5),dtype='float64')
        self.cv_params['C'] = 10**np.asarray(range(-4,4),dtype='float64')
        self.cv_params['C2'] = 10**np.asarray(range(-4,4),dtype='float64')

        #self.cv_params['C'] = np.asarray([1])
        #sself.cv_params['C2'] = np.asarray([.001])

        self.source_learner = method.NadarayaWatsonMethod(configs)
        self.source_learner.quiet = True
        self.base_learner = None
        self.metric = self.configs.metric
        self.opt_succeeded = True

    def _prepare_data(self, data, include_unlabeled=True):
        target_labels = self.configs.target_labels
        data_copy = data.get_transfer_subset(target_labels,include_unlabeled=include_unlabeled)
        #data_copy = data.get_with_labels(target_labels)
        return data_copy

    def get_predictions(self, target_data):
        assert target_data.is_regression
        o = self.source_learner.predict(target_data)
        is_labeled = target_data.is_labeled
        y_s = array_functions.vec_to_2d(o.fu[is_labeled])
        y_true = array_functions.vec_to_2d(o.true_y[is_labeled])
        return (y_s, y_true)

    def train_and_test(self, data):
        assert data.is_regression
        if data.is_regression:
            source_data = data.get_transfer_subset(self.configs.source_labels.ravel(),include_unlabeled=False)
        else:
            source_data = data.get_transfer_subset(self.configs.source_labels.ravel(),include_unlabeled=False)
        source_data.set_target()
        source_data.set_train()
        source_data.reveal_labels(~source_data.is_labeled)
        if source_data.is_regression:
            source_data.data_set_ids[:] = self.configs.target_labels[0]
        if not data.is_regression:
            source_data.change_labels(self.configs.source_labels,self.configs.target_labels)
            source_data = source_data.rand_sample(.1)

        self.source_learner.train_and_test(source_data)

        data_copy = data.get_transfer_subset(self.configs.labels_to_keep, include_unlabeled=True)
        is_source = array_functions.find_set(data_copy.data_set_ids,self.configs.source_labels)
        data_copy.type[is_source] = data_lib.TYPE_SOURCE
        assert data_copy.is_source.any()
        return super(SMSTransfer, self).train_and_test(data_copy)

    def train(self, data):
        assert data.is_regression
        y_s, y_true = self.get_predictions(data)
        I_target = data.is_target
        I_target_labeled = data.is_target & data.is_labeled & data.is_train
        y_s = data.y[I_target_labeled]
        y_true = data.true_y[I_target_labeled]

        x = array_functions.standardize(data.x)
        x_t = x[I_target]
        x_tl = x[I_target_labeled]

        C = self.C
        C2 = self.C2

        W_ll = array_functions.make_rbf(x_tl, self.sigma, self.metric)
        W_ll_reg_inv = np.linalg.inv(W_ll+C2*np.eye(W_ll.shape[0]))
        W_ul = array_functions.make_rbf(x_t, self.sigma, self.metric, x2=x_tl)
        R_ll = W_ll.dot(W_ll_reg_inv)
        R_ul = W_ul.dot(W_ll_reg_inv)
        assert not array_functions.has_invalid(R_ll)
        assert not array_functions.has_invalid(R_ul)
        reg = lambda gh: SMSTransfer.reg(gh, R_ul)
        #f = lambda gh: SMSTransfer.eval(gh, R_ll, R_ul, y_s, y_true, C, reg)
        f = SMSTransfer.eval
        jac = SMSTransfer.gradient

        g0 = np.zeros((R_ll.shape[0] * 2, 1))
        gh_ids = np.zeros(g0.shape)
        gh_ids[R_ll.shape[0]:] = 1

        maxfun = np.inf
        maxitr = np.inf
        constraints = []
        options = {
            'disp': False,
            'maxiter': maxitr,
            'maxfun': maxfun
        }
        method = 'L-BFGS-B'
        #R_ll = np.eye(R_ll.shape[0])
        #R_ul = np.eye(R_ll.shape[0])
        #y_s = 1*np.ones(y_s.shape)
        #y_true = 1*np.ones(y_s.shape)
        args = (R_ll, R_ul, y_s, y_true, C, reg)
        results = optimize.minimize(
            f,
            g0,
            method=method,
            jac=jac,
            options=options,
            constraints=constraints,
            args=args
        )
        check_results = False
        if check_results:
            results2 = optimize.minimize(
                f,
                g0,
                method=method,
                jac=None,
                options=options,
                constraints=constraints,
                args=args
            )
            print self.params
            scipy_opt_methods.compare_results(results, results2, gh_ids)
            diff = results.x-results2.x
            print results.x
            print results2.x
        g, h = SMSTransfer.unpack_gh(results.x, R_ll.shape[0])
        self.opt_succeeded = results.success
        if not results.success:
            print 'SMS Opt failed'

        data.R_ul = R_ul
        self.g = g
        self.h = h
        #assert results.success
        pass

    def predict(self, data):
        o = self.source_learner.predict(data)
        I_target = data.is_target
        if self.opt_succeeded:
            assert not array_functions.has_invalid(self.g)
            assert not array_functions.has_invalid(self.h)
            b = data.R_ul.dot(self.h)
            w = data.R_ul.dot(self.g)
            y_old = o.fu[I_target]
            y_new = (y_old - b) / w
            I_invalid = array_functions.is_invalid(y_new)
            y_new[I_invalid] = y_old[I_invalid]
            o.fu[I_target] = y_new
            o.y[I_target] = y_new
            o.b = b
            o.w = w
        else:
            o.b = np.zeros(I_target.sum())
            o.w = np.ones(I_target.sum())
        o.x = data.x[I_target,:]
        o.assert_input()
        return o

    @staticmethod
    def unpack_gh(gh, n):
        g = gh[0:n]
        h = gh[n:]
        return g,h

    @staticmethod
    def pack_gh(g,h):
        assert g.shape == h.shape
        n = g.shape[0] + h.shape[0]
        gh = np.zeros(n)
        gh[:g.shape[0]] = g
        gh[g.shape[0]:] = h
        return gh

    @staticmethod
    def reg(gh, R_ul):
        g,h = SMSTransfer.unpack_gh(gh, R_ul.shape[1])
        Rg = R_ul.dot(g)
        RRg = R_ul.T.dot(Rg)
        R1 = R_ul.T.dot(np.ones(R_ul.shape[0]))
        grad_g = RRg - R1
        grad_h = np.zeros(h.shape)
        grad = SMSTransfer.pack_gh(grad_g, grad_h)
        grad = grad*2
        return norm(Rg-1)**2, grad

    @staticmethod
    def error(gh, R_ll, y_s, y_true):
        g,h = SMSTransfer.unpack_gh(gh, R_ll.shape[0])
        #g[:] = 0
        #h[:] = 0
        y_d = R_ll.dot(multiply(g, y_true) + h)
        diff = y_d - y_s
        err = norm(diff)**2

        RR = R_ll.T.dot(R_ll)
        D_yt = np.diag(y_true)
        RRD = RR.dot(D_yt)
        DRRD = D_yt.T.dot(RRD)
        DR = D_yt.T.dot(R_ll.T)
        Ry = R_ll.T.dot(y_s)
        grad_g = DRRD.dot(g) + RRD.T.dot(h) - DR.dot(y_s)
        grad_h = RR.dot(h) + RRD.dot(g) - Ry
        grad = SMSTransfer.pack_gh(grad_g, grad_h)
        grad = 2*grad
        return err, grad

    @staticmethod
    def eval(gh, R_ll, R_ul, y_s, y_true, C, reg):
        err_value = SMSTransfer.error(gh, R_ll, y_s, y_true)[0]
        reg_value = reg(gh)[0]
        val = err_value + C*reg_value
        #val = C*reg_value
        #val = err_value
        return val

    @staticmethod
    def gradient(gh, R_ll, R_ul, y_s, y_true, C, reg):
        err_grad = SMSTransfer.error(gh, R_ll, y_s, y_true)[1]
        reg_grad = reg(gh)[1]
        grad = err_grad + C*reg_grad
        #grad = C*reg_grad
        #grad = err_grad
        return grad

    @property
    def prefix(self):
        s = 'SMS'
        return s




