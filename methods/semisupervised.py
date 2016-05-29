
from methods import method
from sklearn.metrics import pairwise
from sklearn.preprocessing import StandardScaler
from utility import array_functions
from utility import helper_functions
import numpy as np

from results_class.results import Output

class SemisupervisedMethod(method.Method):
    def __init__(self, configs=None):
        super(SemisupervisedMethod, self).__init__(configs)
        self.metric = 'euclidean'
        self.graph_transform = StandardScaler()

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
        return L.todense()

class LaplacianRidgeMethod(SemisupervisedMethod):
    def __init__(self, configs=None):
        super(LaplacianRidgeMethod, self).__init__(configs)
        self.cv_params['C'] = self.create_cv_params(-6,6)
        self.cv_params['C2'] = self.create_cv_params(-6,6)
        self.cv_params['sigma'] = self.create_cv_params(-6,6)
        self.transform = StandardScaler()
        self.b = None
        self.w = None


    def train(self, data):
        x_transformed = self.graph_transform.fit_transform(data.x)
        L = self.create_laplacian(x_transformed)


        I = data.is_train & data.is_labeled
        x = data.x[I,:]
        n = x.shape[0]
        p = x.shape[1]
        y = data.y[I]
        #self.b = y.mean()

        x_labeled_transform = self.transform.fit_transform(x)
        x_all_transform = self.transform.transform(data.x)

        x_bias = np.hstack((x_labeled_transform,np.ones((n,1))))
        x_all_bias = np.hstack((x_all_transform,np.ones((x_all_transform.shape[0],1))))
        O = np.eye(p+1)
        O[p,p] = 0
        XX = x_bias.T.dot(x_bias)
        XLX = x_all_bias.T.dot(L).dot(x_all_bias)
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