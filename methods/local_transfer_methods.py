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
import transfer_methods
from copy import deepcopy
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
        self.train_source_learner = True
        self.source_loo = False

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
        o = self.target_learner[0].predict_loo(target_data)
        idx = 0
        is_labeled = target_data.is_labeled
        if self.separate_target_domains:
            is_labeled = np.zeros(target_data.is_labeled.sum(), dtype=np.int)
            for i, label in enumerate(self.configs.target_labels):
                I = (target_data.data_set_ids == label) & target_data.is_labeled
                data_i = target_data.get_subset(I)
                o_i = self.target_learner[i].predict_loo(data_i)
                o.y[idx:idx+data_i.n] = o_i.y
                o.fu[idx:idx+data_i.n] = o_i.fu
                o.true_y[idx:idx+data_i.n] = o_i.true_y
                is_labeled[idx:idx+data_i.n] = I.nonzero()[0]
                idx += data_i.n
        if self.source_loo:
            o_source = self.source_learner.train_predict_loo(target_data)
        else:
            o_source = self.source_learner.predict(target_data)
        if target_data.is_regression:
            y_t = array_functions.vec_to_2d(o.fu)
            if self.source_loo:
                y_s = array_functions.vec_to_2d(o_source.fu)
            else:
                y_s = array_functions.vec_to_2d(o_source.fu[is_labeled])
            y_true = array_functions.vec_to_2d(o.true_y)
        else:
            assert False, 'Update this?'
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

        #Because source learner is probably fully labeled, make sure we're not using validation parameter tuning
        self.source_learner.configs.use_validation = False
        viz_mds = False
        if viz_mds:
            source_labels = self.configs.source_labels
            target_labels = self.configs.target_labels
            data.data_set_ids[:] = 0
            data.data_set_ids[array_functions.find_set(data.y,source_labels[0,:])] = 1
            data.data_set_ids[array_functions.find_set(data.y,source_labels[1,:])] = 2
            data.change_labels(source_labels,target_labels)
            array_functions.plot_MDS(data.x,data.true_y,data.data_set_ids)

        if self.train_source_learner:
            if self.use_stacking:
                self.source_learner.train_and_test(data)
            else:
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











class LocalTransferDelta(HypothesisTransfer):
    def __init__(self, configs=None):
        super(LocalTransferDelta, self).__init__(configs)
        self.base_learner = None
        self.should_plot_g = False
        self.use_oracle = False
        self.quiet = False
        self.separate_target_domains = getattr(configs, 'separate_target_domains', False)
        self.multitask = getattr(configs, 'multitask', False)
        self.C = 0
        self.C2 = 0
        self.C3 = None
        self.reg_MT = 0
        self.sigma_g_learner = None
        self.radius = None
        self.cv_params = {}
        self.cv_params['radius'] = np.asarray([.05, .1, .2],dtype='float64')
        vals = [0] + list(range(-5,5))
        vals.reverse()
        self.cv_params['C'] = 10**np.asarray(vals,dtype='float64')
        self.cv_params['C3'] = np.asarray([0, .2, .4, .6, .8, 1])

        hard_coded_reg_params = False
        if hard_coded_reg_params:
            self.cv_params['C3'] = np.asarray([0])
            self.cv_params['C'] = np.asarray([30])
        self.cv_params['sigma_g_learner'] = self.create_cv_params(-5, 5)
        configs = deepcopy(configs)
        configs.use_validation = False
        self.use_knn = False
        self.use_fused_lasso = getattr(configs, 'use_fused_lasso', False)
        self.target_learner = [method.NadarayaWatsonMethod(deepcopy(configs))]
        self.source_learner = method.NadarayaWatsonMethod(deepcopy(configs))
        use_linear = False
        if self.use_knn:
            self.target_learner = [method.NadarayaWatsonKNNMethod(deepcopy(configs))]
            self.source_learner = method.NadarayaWatsonKNNMethod(deepcopy(configs))
        if use_linear:
            self.target_learner = [method.SKLRidgeRegression(deepcopy(configs))]
            self.source_learner = method.SKLRidgeRegression(deepcopy(configs))
        self.use_l2 = True

        self.source_loo = False
        self.use_stacking = False
        if self.use_stacking:
            self.train_source_learner = True
            self.source_learner = transfer_methods.StackingTransfer(deepcopy(configs))
        else:
            self.source_learner = method.NadarayaWatsonMethod(deepcopy(configs))
            if self.use_knn:
                self.source_learner = method.NadarayaWatsonKNNMethod(deepcopy(configs))

        #self.g_learner = [delta_transfer.CombinePredictionsDelta(deepcopy(configs))]
        self.g_learner = delta_transfer.CombinePredictionsDelta(deepcopy(configs))
        if self.multitask:
            self.g_learner = delta_transfer.CombinePredictionsDeltaMultitask(deepcopy(configs))
        self.g_learner.quiet = True
        self.g_learner.use_l2 = self.use_l2
        self.g_learner.use_fused_lasso = getattr(configs, 'use_fused_lasso', False)

        self.metric = configs.metric
        self.quiet = False
        self.no_C3 = getattr(configs, 'no_C3', False)
        self.constant_b = getattr(configs, 'constant_b', False)
        self.use_radius = getattr(configs, 'use_radius', False)
        self.linear_b = getattr(configs, 'linear_b', False)
        self.clip_b = getattr(configs, 'clip_b', False)
        if self.constant_b:
            del self.cv_params['radius']
            del self.cv_params['C']
            del self.cv_params['sigma_g_learner']
        elif self.linear_b:
            del self.cv_params['radius']
            del self.cv_params['sigma_g_learner']
        if self.multitask:
            assert self.linear_b
            del self.cv_params['C']
            self.cv_params['reg_MT'] = self.create_cv_params(-5, 5)
            self.C = 1e-3
        if not self.use_radius:
            if 'radius' in self.cv_params:
                del self.cv_params['radius']
        if self.no_C3:
            del self.cv_params['C3']
            self.C3 = 0

    def train_g_learner(self, target_data):
        self.g_learner.reg_MT = self.reg_MT
        self.g_learner.C3 = self.C3
        self.g_learner.use_radius = self.use_radius
        self.g_learner.sigma = self.sigma_g_learner
        self.g_learner.C = self.C
        self.g_learner.C2 = self.C2
        self.g_learner.radius = self.radius
        self.g_learner.cv_params = {}
        target_data = target_data.get_subset(target_data.is_train)
        y_t, y_s, y_true = self.get_predictions(target_data)

        is_labeled = target_data.is_labeled
        if target_data.is_regression:
            a = y_s - y_t
            b = y_t - y_true
        else:
            a = y_s[:, 0] - y_t[:, 0]
            b = y_t[:, 0] - y_true[:, 0]

        parametric_data = target_data.get_subset(is_labeled)
        # parametric_data = target_data
        parametric_data.a = a
        parametric_data.b = b
        parametric_data.y_s = y_s
        parametric_data.y_t = y_t
        parametric_data.set_target()
        parametric_data.set_train()
        parametric_data.data_set_ids = target_data.data_set_ids[is_labeled]
        # assert target_data.is_regression
        '''
        if target_data.is_regression:
            parametric_data.data_set_ids[:] = self.configs.target_labels[0]
        '''
        # s = np.hstack((a,b))
        # s[parametric_data.x.argsort(0)]
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

    def train_and_test(self, data):
        target_labels = self.configs.target_labels
        num_target_domains = target_labels.size
        assert len(self.target_learner) == 1
        if self.separate_target_domains:
            for i in range(num_target_domains):
                if i > 0:
                    self.target_learner.append(deepcopy(self.target_learner[0]))
                target_data = data.get_transfer_subset(target_labels[i], include_unlabeled=True)
                assert target_data.n > 0
                self.target_learner[i].train_and_test(target_data)
        else:
            target_data = self.get_target_subset(data)
            self.target_learner[0].train_and_test(target_data)
        if self.use_stacking:
            assert num_target_domains == 1
        if self.use_stacking:
            self.source_learner.train_and_test(data)
        results = super(LocalTransferDelta, self).train_and_test(data)
        # print self.g_learner.g
        '''
        self.plot_g()
        I = np.squeeze(self.g_learner.g_nw.x.argsort(0))
        sorted_x = self.g_learner.g_nw.x[I,:]
        sorted_g = self.g_learner.g_nw.y[I]
        print sorted_x.T
        print sorted_g
        print self.g_learner.g_nw.sigma
        '''
        print_sparse_g = False
        if print_sparse_g:
            g = self.g_learner.g
            g[np.abs(g) < 1e-5] = 0
            for name, gi in zip(data.feature_names, g):
                print name + ': ' + str(gi)
        return results

    def train(self, data):
        target_labels = self.configs.target_labels
        if self.separate_target_domains:
            for i, label in enumerate(target_labels):
                target_data = data.get_transfer_subset(target_labels[i], include_unlabeled=True)
                self.target_learner[i].train(target_data)
        else:
            target_data = self.get_target_subset(data)
            self.target_learner[0].train(target_data)
        target_data = self.get_target_subset(data)
        self.train_g_learner(target_data)
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
        o = self.target_learner[0].predict(data)
        if self.separate_target_domains:
            for i, label in enumerate(self.configs.target_labels):
                I = data.data_set_ids == label
                o_i = self.target_learner[i].predict(data)
                o.y[I] = o_i.y[I]
                o.fu[I] = o_i.fu[I]
        is_target = data.is_target
        o_source = self.source_learner.predict(data.get_subset(is_target))
        if not data.is_regression:
            assert o.fu.ndim == 2
        else:
            assert np.squeeze(o.fu).ndim == 1
            assert np.squeeze(o_source.fu).ndim == 1
            o.fu = o.fu.reshape((o.fu.size,1))
            o_source.fu = o_source.fu.reshape((o_source.fu.size,1))
        for i in range(o.fu.shape[1]):
            fu_t = o.fu[is_target,i]
            fu_s = o_source.fu[:,i]
            if self.g_learner is not None:
                pred = self.g_learner.combine_predictions(data.x[is_target,:],fu_s,fu_t, data.data_set_ids[is_target])
                if data.x.shape[1] == 1 and not self.multitask:
                    x = scipy.linspace(data.x.min(),data.x.max(),100)
                    x = array_functions.vec_to_2d(x)
                    g = self.g_learner.predict_g(x, data.data_set_ids)
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
        '''
        if data.x.shape[1] == 1 and not self.multitask:
            x = array_functions.vec_to_2d(scipy.linspace(data.x.min(),data.x.max(),100))
            o.linspace_x = x
            o.linspace_g = self.g_learner.predict_g(x)
        '''
        assert not (np.isnan(o.y)).any()
        assert not (np.isnan(o.fu)).any()
        return o

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
        if getattr(self, 'use_radius', False) or not is_nonparametric:
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
        if getattr(self, 'use_stacking', False):
            s += '-stacking'
        if getattr(self, 'source_loo', False):
            s += '-sourceLOO'
        if getattr(self, 'use_knn', False):
            s += '_knn'
        if getattr(self, 'separate_target_domains', False):
            s += '_sep-target'
        if getattr(self, 'multitask', False):
            s += '_multitask'
        #s += '-TESTING_REFACTOR'
        return s



def pack_v(ft, b, alpha):
    #return np.concatenate((b, alpha, ft))
    return np.concatenate((b, alpha))

def unpack_v(v, n, p, opt_data):
    #assert v.size / 2 == n
    b_end = n
    if opt_data['linear_b']:
        b_end = p + 1
    b = v[:b_end]
    alpha = v[b_end:b_end+n]
    ft = v[b_end+n:]
    if ft.size == 0:
        ft = None
    return b, alpha, ft

from numpy.linalg import norm

def f_delta_new(alpha, b, ft, y_s, S_b, S_a):
    a_smooth = S_a.dot(alpha)
    f = (1 - a_smooth) * ft + a_smooth * (b + y_s)
    return f

check_gradient_new = False
from timer.timer import tic, toc
def gradient_delta_new(v, opt_data):
    n, p = opt_data['x'].shape

    b, alpha, ft = unpack_v(v, n, p, opt_data)
    learn_ft = True
    if ft is None:
        learn_ft = False
        ft = opt_data['y_t']
    y = opt_data['y']
    x = opt_data['x']
    y_s = opt_data['y_s']
    S_x = opt_data['S_x']
    if not learn_ft:
        S_x = np.eye(n)
    S_b = opt_data['S_b']
    S_a = opt_data['S_a']
    C_ft = opt_data['C_ft']
    C_alpha = opt_data['C_alpha']
    C_b = opt_data['C_b']
    A = opt_data['A']
    linear_b = opt_data['linear_b']

    if linear_b:
        b_w = b[:p]
        b_c = b[-1]
        b_pred = x.dot(b_w) + b_c
    else:
        b_pred = S_b.dot(b)
    A = S_x.dot(ft)[:, None] * S_a
    B = (b_pred + y_s)[:, None] * S_a
    C = S_x.dot(ft) - y
    D = B - A
    da = 2 * (D.T.dot(D.dot(alpha)) + D.T.dot(C))


    if linear_b:
        S_aa = S_a.dot(alpha)
        S_ft = S_x.dot(ft)
        M_a = (1-S_aa)*S_ft + S_aa*y_s - y
        M_b = S_aa[:, None] * x
        dw = 2*(M_b.T.dot(M_b.dot(b_w)) + M_b.T.dot(M_a) + M_b.T.dot(S_aa)*b_c + C_b*b_w )
        dc = 2*(M_a.T.dot(S_aa) + b_w.T.dot(M_b.T).dot(S_aa) + S_aa.T.dot(S_aa)*b_c)
        db = np.append(dw, dc)

        f = f_delta_new(alpha, x.dot(b_w) + b_c, S_x.dot(ft), y_s, S_b, S_a)
        loss_f = norm(f - y) ** 2 + C_b * norm(b_w) ** 2
        loss_f2 = norm(M_a + M_b.dot(b_w) + S_aa*b_c)**2 + C_b*norm(b_w)**2
        print ''

    else:
        M_a = (1 - S_a.dot(alpha)) * S_x.dot(ft) + (S_a.dot(alpha)) * y_s - y
        M_b = S_a.dot(alpha)[:, None] * S_b
        db = 2 * (M_b.T.dot(M_b.dot(b)) + M_b.T.dot(M_a))
    g = np.concatenate((db, da))
    if learn_ft:
        print 'TODO: Accelerate this!'
        G= np.diag(1 - S_a.dot(alpha)).dot(S_x)
        F = S_a.dot(alpha) * (S_b.dot(b) + y_s) - y
        df = 2*(G.T.dot(G).dot(ft) + G.T.dot(F))
        g = np.concatenate(g, df)
    if check_gradient_new:
        g_approx = optimize.approx_fprime(v, lambda x: eval_delta_new(x, opt_data), 1e-8)
        rel_err = norm(g[:2]-g_approx[:2])/norm(g_approx[:2])
        print 'rel err: ' + str(rel_err)
        #rel_err_a = norm(g[20:]-g_approx[20:])/norm(g_approx[20:])
        rel_err_b = norm(g[:2] - g_approx[:2]) / norm(g_approx[:2])
        '''
        if np.isfinite(rel_err_a) and rel_err_a > 1e-4:
            print 'Grad error alpha: ' + str(rel_err_a)
        if np.isfinite(rel_err_b) and rel_err_b > 1e-4:
            print 'Grad error b: ' + str(rel_err_b)
        '''
        #assert
    return g

def eval_delta_new(v, opt_data):
    n, p = opt_data['x'].shape
    b, alpha, ft = unpack_v(v, n, p, opt_data)
    learn_ft = True
    if ft is None:
        ft = opt_data['y_t']
        learn_ft = False
    y = opt_data['y']
    x = opt_data['x']
    y_s = opt_data['y_s']
    S_x = opt_data['S_x']
    if not learn_ft:
        S_x = np.eye(n)
    S_b = opt_data['S_b']
    S_a = opt_data['S_a']
    C_ft = opt_data['C_ft']
    C_b = opt_data['C_b']
    C_alpha = opt_data['C_alpha']
    A = opt_data['A']
    linear_b = opt_data['linear_b']

    if linear_b:
        b_w = b[:p]
        b_c = b[-1]
        f = f_delta_new(alpha, x.dot(b_w) + b_c, S_x.dot(ft), y_s, S_b, S_a)
    else:
        f = f_delta_new(alpha, S_b.dot(b), S_x.dot(ft), y_s, S_b, S_a)

    loss_f = norm(f - y)**2
    if linear_b:
        loss_f += C_b * norm(b_w)**2
    if check_gradient_new:
        pass
    '''
    loss_ft = norm(ft - S_x.dot(y))**2
    loss_a = norm(S_x.dot(alpha) - A)**2
    return loss_f + C_ft*loss_ft + C_alpha*loss_a
    '''
    return loss_f

from sklearn.preprocessing import StandardScaler
class LocalTransferDeltaNew(LocalTransferDelta):
    def __init__(self, configs=None):
        super(LocalTransferDelta, self).__init__(configs)

        self.quiet = False
        self.cv_params = {}
        self.cv_params['sigma_target'] = self.create_cv_params(-5, 5)
        self.cv_params['sigma_b'] = self.create_cv_params(-5, 5)
        self.cv_params['sigma_alpha'] = self.create_cv_params(-5, 5)
        #self.cv_params['C_ft'] = self.create_cv_params(-5, 5, append_zero=True)
        #self.cv_params['C_alpha'] = self.create_cv_params(-5, 5, append_zero=True)
        #self.cv_params['A'] = np.asarray([0, .25, .5, .75, 1])
        self.sigma_target = 1
        self.sigma_b = 1
        self.sigma_alpha = 1
        self.A = 0
        self.C_alpha = 0
        self.C_b = 0
        self.C_ft = 0

        self.sigma_alpha = 1
        self.use_grad = True
        self.use_bounds = True
        self.optimize_ft = False
        self.linear_b = False

        if self.linear_b:
            del self.cv_params['sigma_b']
            self.cv_params['C_b'] = self.create_cv_params(-5, 5, append_zero=True)

        configs = deepcopy(configs)
        #configs.use_validation = False
        self.source_learner = method.NadarayaWatsonMethod(deepcopy(configs))
        self.base_learner = method.NadarayaWatsonMethod(deepcopy(configs))
        self.g_learner = None

        self.metric = configs.metric
        self.quiet = False
        self.train_source_learner = True
        self.use_stacking = False
        self.transform =  StandardScaler()


    def train_and_test(self, data):
        target_labels = self.configs.target_labels
        #source_data = data.get_transfer_subset(self.configs.source_labels, include_unlabeled=False)
        #self.source_learner.train_and_test(source_data)
        results = super(LocalTransferDelta, self).train_and_test(data)
        return results




    def train(self, data):

        target_data_all = self.get_target_subset(data)
        target_data_labeled = target_data_all.get_subset(target_data_all.is_train & target_data_all.is_labeled)
        y_s = self.source_learner.predict(target_data_labeled).y
        y = target_data_labeled.true_y
        x = target_data_labeled.x
        if self.transform is not None:
            x = self.transform.fit_transform(x)

        W_x = array_functions.make_rbf(x, self.sigma_target)
        S_x = array_functions.make_smoothing_matrix(W_x)
        y_t = S_x.dot(y)
        W_b = array_functions.make_rbf(x, self.sigma_b)
        S_b = array_functions.make_smoothing_matrix(W_b)
        W_a = array_functions.make_rbf(x, self.sigma_alpha)
        S_a = array_functions.make_smoothing_matrix(W_a)
        C_ft = self.C_ft
        C_alpha = self.C_alpha

        A = self.A
        #S_a = S_x = S_b = np.eye(target_data_labeled.n)
        '''
        n = target_data_labeled.n
        b_translate = cvx.Variable(n)
        ft = cvx.Variable(n)
        ft_loss = cvx.sum_squares(ft - S_x.dot(y))
        b_fs = y_s + S_b * b_translate
        #b_loss = cvx.sum_squares(1 - S_b.dot(b))
        f_loss = cvx.sum_squares(.5*ft + .5*b_fs - y)
        total_loss = ft_loss + C*ft_loss
        '''
        opt_data = {
            'S_x': S_x,
            'S_b': S_b,
            'S_a': S_a,
            'C_ft': C_ft,
            'C_alpha': C_alpha,
            'y': y,
            'y_s': y_s,
            'x': x,
            'A': A,
            'y_t': y_t,
            'linear_b': self.linear_b,
            'C_b': self.C_b,
        }

        f = lambda v: eval_delta_new(v, opt_data)
        g = lambda v: gradient_delta_new(v, opt_data)
        p = x.shape[1]
        b_size = y.size
        if self.linear_b:
            b_size = p + 1
        f0 = np.zeros(y.size + b_size)
        if self.optimize_ft:
            f0 = np.zeros(2*y.size + b_size)
        if self.linear_b:
            assert not self.optimize_ft

        bounds = None
        if self.use_bounds:
            bounds = [(None, None) for i in range(b_size)] + [(0, 1 ) for i in range(y.size)]
            if self.optimize_ft:
                bounds += [(None, None) for i in range(y.size)]

        if self.use_grad:
            results = optimize.minimize(
                f,
                f0,
                method=None,
                jac=g,
                options=None,
                constraints=None,
                bounds=bounds,
            )

        if not self.use_grad:
            results = optimize.minimize(
                f,
                f0,
                method=None,
                jac=None,
                options=None,
                constraints=None,
                bounds=bounds,
            )


        #rel_err = norm(results.x - results2.x)/norm(results.x)

        b, alpha, ft = unpack_v(results.x, y.size, x.shape[1], opt_data)
        if ft is None:
            ft = y_t
        self.b = b
        self.alpha = alpha
        self.ft = ft
        self.y = y
        self.x = x


    def predict(self, data):
        o = self.source_learner.predict(data)
        y_s = o.y
        x = self.x
        x2 = data.x
        if self.transform is not None:
            x2 = self.transform.transform(x2)
        W_x = array_functions.make_rbf(x, self.sigma_target, x2=x2).T
        S_x = array_functions.make_smoothing_matrix(W_x)
        ft = S_x.dot(self.y)
        W_b = array_functions.make_rbf(x, self.sigma_b, x2=x2).T
        S_b = array_functions.make_smoothing_matrix(W_b)
        W_a = array_functions.make_rbf(x, self.sigma_alpha, x2=x2).T
        S_a = array_functions.make_smoothing_matrix(W_a)

        if self.linear_b:
            b = data.x.dot(self.b[:-1]) + self.b[-1]
        else:
            b = S_b.dot(self.b)
        f = f_delta_new(self.alpha, b, ft, y_s, S_b, S_a)
        o.y = f.copy()
        o.fu = f.copy()
        return o

    @property
    def prefix(self):
        s = 'LocalTransferNew'
        if getattr(self, 'use_grad'):
            s += '-grad'
        if getattr(self, 'use_bounds'):
            s += '-bounds'
        if getattr(self, 'optimize_ft'):
            s += '-opt_ft'
        if getattr(self, 'linear_b'):
            s += '-linearB'
        if getattr(self.configs, 'use_validation', False):
            s += '-VAL'
        return s



class OffsetTransfer(HypothesisTransfer):
    def __init__(self, configs=None):
        super(OffsetTransfer, self).__init__(configs)
        self.cv_params = dict()
        self.g_learner = method.NadarayaWatsonMethod(configs)
        self.x_source = None
        self.y_source_new = None
        self.joint_cv = True
        if self.joint_cv:
            self.cv_params['sigma_g_learner'] = 10**np.asarray(range(-5,5),dtype='float64')
            self.cv_params['sigma_target_learner'] = 10 ** np.asarray(range(-5, 5), dtype='float64')


    def train_and_test(self, data):
        source_data = self.get_source_data(data)
        self.source_learner.configs.use_validation = False
        self.source_learner.train_and_test(source_data)
        '''
        data_copy = self._prepare_data(data, include_unlabeled=True)
        data_copy = data_copy.get_transfer_subset(self.configs.target_labels, include_unlabeled=True)
        data_copy = data_copy.get_subset(data_copy.is_target)
        '''
        data_copy = copy.deepcopy(data)
        return super(HypothesisTransfer, self).train_and_test(data_copy)

    def train(self, data):
        if self.joint_cv:
            self.g_learner.sigma = self.sigma_g_learner
            self.target_learner.sigma = self.sigma_target_learner
        target_data = self.get_target_subset(data)
        #target_data = target_data.get_subset(target_data.is_labeled)
        offset_data = deepcopy(target_data)
        offset_data.y -= self.source_learner.predict(offset_data).y
        if self.joint_cv:
            self.g_learner.train(offset_data)
        else:
            self.g_learner.train_and_test(offset_data)
        source_data = self.get_source_data(data)
        x_source = source_data.x
        y_source_new = source_data.y + self.g_learner.predict(source_data).y
        source_data_new = data_lib.Data(x_source, y_source_new)

        all_data = copy.deepcopy(target_data)
        all_data.combine(source_data_new)
        if self.joint_cv:
            self.target_learner.train(all_data)
        else:
            self.target_learner.train_and_test(all_data)

    def predict(self, data):
        o = self.target_learner.predict(data)
        return o

    @property
    def prefix(self):
        s = 'OffsetTransfer'
        if getattr(self, 'joint_cv', False):
            s += '-jointCV'
        if getattr(self.configs, 'use_validation', False):
            s += '-VAL'
        return s

class LocalTransferDeltaSMS(LocalTransferDelta):
    def __init__(self, configs=None):
        super(LocalTransferDeltaSMS, self).__init__(configs)
        self.C2 = 0
        self.include_scale = getattr(configs, 'include_scale', False)
        self.cv_params = {}
        vals = list(range(-7,8))
        vals.reverse()
        #vals = [0,1,2]
        self.cv_params['sigma'] = 10**np.asarray(vals,dtype='float64')
        self.cv_params['C'] = 10**np.asarray(vals,dtype='float64')
        if self.include_scale:
            self.cv_params['C2'] = 10**np.asarray(vals,dtype='float64')
        self.g_learner = delta_transfer.CombinePredictionsDeltaSMS(deepcopy(configs))
        self.g_learner.include_scale = self.include_scale
        self.g_learner.quiet = True
        self.quiet = False
        self.num_splits = 5
        self.use_validation = configs.use_validation


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
        if getattr(self, 'use_validation', False):
            s += '-VAL'
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
        self.use_validation = configs.use_validation

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
        if getattr(self, 'use_validation', False):
            s += '-VAL'
        return s




