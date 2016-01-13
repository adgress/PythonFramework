from methods import method
from utility import array_functions
from configs.base_configs import  MethodConfigs
from results_class import results
import scipy
import sklearn
import numpy as np
from loss_functions import loss_function
import statsmodels.api as sm

class KDE(method.Method):

    def __init__(self,configs=MethodConfigs()):
        super(KDE, self).__init__(configs)
        self.cv_params = {
            'sigma': np.asarray(10.0**np.asarray(range(-4,5)))
        }
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
