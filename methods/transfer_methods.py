__author__ = 'Aubrey'
import method
import copy
from data import data as data_lib
from results_class import results as results_lib
from utility import array_functions
import cvxpy as cvx
import numpy as np
from loss_functions import loss_function
import matplotlib.pyplot as plt

enable_plotting = True

class TargetTranfer(method.Method):
    def __init__(self, configs=None):
        super(TargetTranfer, self).__init__(configs)
        self.base_learner = method.SKLLogisticRegression(configs)
        self.cv_params = {}
        self.base_learner.experiment_results_class = self.experiment_results_class

    def train(self, data):
        self.base_learner.train_and_test(data)

    def train_and_test(self, data):
        data_copy = self._prepare_data(data)
        return super(TargetTranfer, self).train_and_test(data_copy)

    def _prepare_data(self, data):
        target_labels = self.configs.target_labels
        data_copy = data.get_transfer_subset(target_labels,include_unlabeled=False)
        #data_copy = data.get_with_labels(target_labels)
        return data_copy

    def predict(self, data):
        return self.base_learner.predict(data)

    @property
    def prefix(self):
        return 'TargetTransfer+' + self.base_learner.prefix

class FuseTransfer(TargetTranfer):
    def __init__(self, configs=None):
        super(FuseTransfer, self).__init__(configs)

    def _prepare_data(self, data):
        source_labels = self.configs.source_labels
        target_labels = self.configs.target_labels
        data_copy = copy.deepcopy(data)
        #source_inds = array_functions.find_set(data_copy.true_y,source_labels)
        source_inds = data.get_transfer_inds(source_labels)
        if not data_copy.is_regression:
            data_copy.change_labels(source_labels,target_labels)
        data_copy.type[source_inds] = data_lib.TYPE_SOURCE
        data_copy = data_copy.get_transfer_subset(np.concatenate((source_labels,target_labels)),include_unlabeled=True)
        data_copy.is_train[data_copy.is_source] = True
        data_copy.reveal_labels(data_copy.is_source)
        return data_copy

    @property
    def prefix(self):
        return 'FuseTransfer+' + self.base_learner.prefix

class LocalTransfer(FuseTransfer):
    def __init__(self, configs=None):
        super(LocalTransfer, self).__init__(configs)
        #self.cv_params['sigma'] = 10**np.asarray(range(-4,4),dtype='float64')
        self.sigma = 100
        #self.cv_params['radius'] = np.asarray([.01, .05, .1, .15, .2],dtype='float64')
        self.cv_params['C'] = 10**np.asarray(range(-4,4),dtype='float64')
        self.k = 3
        #self.C = 10
        #self.radius = .05
        #self.cv_params['C'] = np.asarray([.00000001,.0000001,.000000001,.0000000001,.000001],dtype='float64')
        self.target_learner = method.NadarayaWatsonMethod(configs)
        self.source_learner = method.NadarayaWatsonMethod(configs)
        self.base_learner = None
        self.learn_g_just_labeled = True
        self.should_plot_g = False
        self.use_fused_lasso = True

    def train(self, data):
        #self.C = .00000001
        #self.C = 0
        #x = cvx.Variable()
        #y = cvx.Variable()
        #constraints = [x + y == 1, x - y >= 1]

        assert not data.is_regression, 'Confirm this works!'

        if data.is_regression:
            target_data = data.get_transfer_subset(self.configs.target_labels, include_unlabeled=True)
        else:
            target_data = data.get_subset(data.is_target)

        o = self.target_learner.predict_loo(target_data)
        o_source = self.source_learner.predict(target_data)
        is_labeled = target_data.is_labeled

        target_labels = self.configs.target_labels
        if data.is_regression:
            y_t = array_functions.vec_to_2d(o.fu)
            y_s = array_functions.vec_to_2d(o_source.fu)
            y_true = array_functions.vec_to_2d(o.true_y)
        else:
            y_t = o.fu[:,target_labels]
            y_s = o_source.fu[:,target_labels]
            y_s = y_s[is_labeled,:]
            y_true = array_functions.make_label_matrix(o.true_y)[:,target_labels]
            y_true = array_functions.try_toarray(y_true)



        labeled_inds = is_labeled.nonzero()[0]
        unlabeled_inds = (~is_labeled).nonzero()[0]

        n_labeled = len(labeled_inds)
        metric='euclidean'
        #metric='cosine'
        if self.learn_g_just_labeled:

            g = cvx.Variable(n_labeled)
            '''
            L = array_functions.make_laplacian_uniform(target_data.x[labeled_inds,:],self.radius,metric) \
                + .0001*np.identity(n_labeled)
            '''
            L = array_functions.make_laplacian_kNN(target_data.x[labeled_inds,:],self.k,metric) \
                + .0001*np.identity(n_labeled)
            if self.use_fused_lasso:
                reg = 0
                inds = L.nonzero()
                rows = np.asarray(inds[0]).T.squeeze()
                cols = np.asarray(inds[1]).T.squeeze()
                for i in range(len(inds[0])):
                    row = rows[i]
                    col = cols[i]
                    if row == col:
                        continue
                    Lij = L[row,col]
                    '''
                    if i >= j or Lij == 0:
                        continue
                    '''
                    reg = reg -  Lij*cvx.abs(g[row]-g[col])


            else:
                reg = cvx.quad_form(g,L)
            if not data.is_regression:
                loss = cvx.sum_entries(
                    cvx.power(
                        cvx.mul_elemwise(y_s[:,0], g) + cvx.mul_elemwise(y_t[:,0], (1-g)) - y_true[:,0],
                        2
                    )
                )
            else:
                loss = cvx.sum_entries(
                    cvx.power(
                        cvx.mul_elemwise(y_s[is_labeled,0], g) + cvx.mul_elemwise(y_t[:,0], (1-g)) - y_true[:,0],
                        2
                    )
                )
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
        else:
            g = cvx.Variable(target_data.n)
            sorted_inds = np.concatenate((labeled_inds,unlabeled_inds))
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
            constraints = [g >= 0, g <= 1]
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

        pass

    def train_and_test(self, data):
        #source_data = data.get_with_labels(self.configs.source_labels)
        if data.is_regression:
            source_data = data.get_transfer_subset(self.configs.source_labels,include_unlabeled=True)
        else:
            source_data = data.get_transfer_subset(self.configs.source_labels,include_unlabeled=False)
        source_data.set_target()
        source_data.set_train()
        source_data.reveal_labels(~source_data.is_labeled)
        if not data.is_regression:
            source_data.change_labels(self.configs.source_labels,self.configs.target_labels)
            source_data = source_data.rand_sample(.1)
        self.source_learner.train_and_test(source_data)
        target_data = data.get_transfer_subset(self.configs.target_labels, include_unlabeled=True)
        self.target_learner.train_and_test(target_data)
        return super(LocalTransfer, self).train_and_test(data)

    def predict(self, data):
        o = self.target_learner.predict(data)
        o2 = self.target_learner.predict(data)
        is_target = data.is_target
        o_source = self.source_learner.predict(data.get_subset(is_target))
        assert o.fu.ndim == 2
        for i in range(o.fu.shape[1]):
            fu_t = o.fu[is_target,i]
            fu_s = o_source.fu[:,i]
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
        return 'LocalTransfer'


