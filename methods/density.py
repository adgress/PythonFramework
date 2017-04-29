from methods import method
from utility import array_functions
from configs.base_configs import  MethodConfigs
from results_class import results
import scipy
import sklearn
import numpy as np
from loss_functions import loss_function
import statsmodels.api as sm
from utility import array_functions

class KDE(method.Method):

    def __init__(self,configs=MethodConfigs()):
        super(KDE, self).__init__(configs)
        self.cv_params = {
            'sigma': np.asarray(10.0**np.asarray(range(-4,5)))
        }
        self.cv_params = {}
        self.is_classifier = False
        self._estimated_error = None
        self.quiet = True
        self.best_params = None
        self.model = None
        self.configs.loss_function = loss_function.MeanSquaredError()
        self.configs.cv_loss_function = loss_function.MeanSquaredError()

    def run_method(self, data):
        self.train(data)
        if data.x.shape[0] == 0:
            assert False
            self.train(data)
        return self.predict(data)

    def train_and_test(self, data):
        return self.run_method(data)

    def train(self, data):
        x = data.x
        x = array_functions.vec_to_2d(x)
        self.model = sm.nonparametric.KDEMultivariate(
            x,
            var_type='c'*x.shape[1],
            bw='cv_ls'
        )

    def predict(self, data):
        x = array_functions.vec_to_2d(data.x)
        x = array_functions.vec_to_2d(x)
        y = self.model.pdf(x)
        o = results.Output(data,y)
        return o


def compute_kernel(x, x_predict, bandwidth):
    if x_predict is None:
        x_predict = x
    D = array_functions.make_graph_distance(x_predict, x) / bandwidth
    use_gaussian = True
    n, p = x.shape
    n = float(n)
    p = float(p)
    if use_gaussian:
        K = np.exp(-.5 * D ** 2)
        #K *= (1 / (x.shape[0] * bandwidth)) * (1 / np.sqrt(2 * np.pi))
        K /= (n * (bandwidth**(p/2)) * (2 * np.pi)**(p/2))
    else:
        K = D
        K[K <= 1] = .5
        K[ K > 1] = 0
        K /= (x.shape[0] * bandwidth)
    return K

def compute_density(X, X_predict, bandwidth):
    if X_predict is None:
        X_predict = X
    K = compute_kernel(X, X_predict, bandwidth)
    return K.sum(1)

def avg_neg_log_likelihood(X, X_predict, bandwidth, loo=False):
    if loo:
        d = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            Xi = np.delete(X, i, axis=0)
            d[i] = compute_density(Xi, X[np.newaxis,i,:], bandwidth, )
    else:
        d = compute_density(X, X_predict, bandwidth)
    try:
        val = -np.log(d).mean()
    except:
        pass
    return val

def get_best_bandwidth(X, bandwidths, X_validation=None):
    vals = np.zeros(bandwidths.size)
    for i, b in enumerate(bandwidths):
        if X_validation is None:
            vals[i] = avg_neg_log_likelihood(X, None, b, loo=True)
        else:
            vals[i] = avg_neg_log_likelihood(X, X_validation, b)
    idx = vals.argmin()
    return bandwidths[idx], vals

if __name__ == '__main__':
    X = np.random.rand(100, 10)
    X2 = np.random.rand(50, 10)
    bandwidths = np.logspace(-3, 3)
    for bandwidth in bandwidths:
        #print str(avg_neg_log_likelihood(X, X2, bandwidth))
        print str(avg_neg_log_likelihood(X, None, bandwidth, loo=True))
    pass