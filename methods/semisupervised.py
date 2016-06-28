
from methods import method
from sklearn.metrics import pairwise
from sklearn.preprocessing import StandardScaler
from utility import array_functions
from utility import helper_functions
import numpy as np
from data import data as data_lib
from copy import deepcopy

from results_class.results import Output

class SemisupervisedMethod(method.Method):
    def __init__(self, configs=None):
        super(SemisupervisedMethod, self).__init__(configs)
        self.metric = 'euclidean'
        self.graph_transform = StandardScaler()
        self.max_n_L = 500

    def create_similarity_matrix(self, x1, x2=None):
        if x2 is None:
            x2 = x1
        W = pairwise.pairwise_distances(x1, x2, self.metric)
        W = np.square(W)
        W = -self.sigma * W
        W = np.exp(W)
        return W

    def create_laplacian(self, x1, x2=None):
        if x2 is None:
            x2 = x1
        W = self.create_similarity_matrix(x1, x2)
        L = array_functions.make_laplacian_with_W(W)
        return np.asarray(L.todense())

class SemisupervisedRegressionMethod(SemisupervisedMethod):
    def __init__(self, configs=None):
        super(SemisupervisedRegressionMethod, self).__init__(configs)
        self.cv_params['C'] = self.create_cv_params(-5, 5)
        self.cv_params['sigma'] = self.create_cv_params(-5, 5)
        self.max_n_L = 200
        self.nw_method = method.NadarayaWatsonMethod(deepcopy(configs))

    def train(self, data):
        I = data.is_train & data.is_labeled
        x = data.x[I, :]
        n_L = x.shape[0]
        y_L = np.expand_dims(data.y[I],1)


        x_L = x
        x_U = data.x
        if x_U.shape[0] > self.max_n_L:
            to_keep = I.nonzero()[0]
            to_sample = (~I).nonzero()[0]
            num_to_sample = self.max_n_L - to_keep.size
            sampled = np.random.choice(to_sample, num_to_sample, replace=False)
            to_use = np.hstack((to_keep, sampled))
            x_U = x_U[to_use, :]

        x_U_not_transformed = x_U
        x_L_not_transformed = x_L
        x_U = self.graph_transform.fit_transform(x_U)


        n_U = x_U.shape[0]
        L_UU = self.create_laplacian(x_U, x_U)
        L_inv = np.linalg.inv(L_UU + self.C*np.eye(n_U))
        D_L_inv = np.diag(1 / L_inv.sum(1))
        S_SSL = D_L_inv.dot(L_inv)
        S_SSL_UL = S_SSL[:, :n_L]
        S_SSL_UL = self.fix_matrix_rows(S_SSL_UL, 1.0/n_L)
        '''
        x_L = self.graph_transform.transform(x_L)
        W_UL = self.create_similarity_matrix(x_U, x_L)
        D_W = np.diag(1 / W_UL.sum(1))
        S_NW = (1-self.C)*D_W.dot(W_UL)
        S_NW = self.fix_matrix_rows(S_NW, 1.0/n_L)
        f = (S_SSL_UL + S_NW).dot(y_L)
        '''
        f = np.squeeze(S_SSL_UL.dot(y_L))
        nw_data = data_lib.Data(x_U_not_transformed, f)
        nw_data.is_regression = True

        #self.nw_method.train_and_test(nw_data)

        I = nw_data.is_labeled
        self.nw_method.x = nw_data.x
        self.nw_method.y = nw_data.y
        self.nw_method.tune_loo(nw_data)

        #self.nw_method.sigma = self.sigma
        #self.nw_method.train(nw_data)

    def fix_matrix_rows(self, S, c):
        sums = S.sum(1)
        should_replace = (sums < 1e-16) | ~np.isfinite(sums)
        S[should_replace, :] = c
        return S

    def predict(self, data):
        o = self.nw_method.predict(data)
        return o

    @property
    def prefix(self):
        return 'SLL-NW'

class LaplacianRidgeMethod(SemisupervisedMethod):
    def __init__(self, configs=None):
        super(LaplacianRidgeMethod, self).__init__(configs)
        self.cv_params['C'] = self.create_cv_params(-5,5)
        self.cv_params['C2'] = self.create_cv_params(-5,5)
        self.cv_params['sigma'] = self.create_cv_params(-5,5)
        self.transform = StandardScaler()
        self.b = None
        self.w = None


    def train(self, data):
        I = data.is_train & data.is_labeled
        x = data.x[I,:]
        n = x.shape[0]
        p = x.shape[1]
        y = data.y[I]

        x_labeled_transform = self.transform.fit_transform(x)
        x_all_transform = self.transform.transform(data.x)

        x_bias = np.hstack((x_labeled_transform,np.ones((n,1))))
        x_all_bias = np.hstack((x_all_transform,np.ones((x_all_transform.shape[0],1))))
        O = np.eye(p+1)
        O[p,p] = 0
        x_L = x_all_bias
        if x_L.shape[0] > self.max_n_L:
            I_L = np.random.choice(x_L.shape[0], self.max_n_L, replace = False)
            x_L = x_L[I_L,:]
        L = self.create_laplacian(x_L)
        XX = x_bias.T.dot(x_bias)
        XLX = x_L.T.dot(L).dot(x_L)
        A = XX + self.C*O + self.C2*XLX
        v = np.linalg.lstsq(A,x_bias.T.dot(y))
        w_anal = array_functions.vec_to_2d(v[0][0:p])
        b_anal = v[0][p]
        self.w = w_anal
        self.b = b_anal


    def predict(self, data):
        o = Output(data)
        x = self.transform.transform(data.x)
        y = x.dot(self.w) + self.b
        o.fu = y
        o.y = y
        return o

    @property
    def prefix(self):
        return 'LapRidge'