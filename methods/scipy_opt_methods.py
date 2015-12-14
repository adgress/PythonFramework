import scipy
from scipy import optimize
from methods import method
from configs import base_configs
from utility import array_functions
import numpy as np
from numpy.linalg import norm
import math
from results_class import results
from data import data as data_lib
from timer.timer import tic,toc

class ScipyOptRidgeRegression(method.Method):
    def __init__(self,configs=base_configs.MethodConfigs()):
        super(ScipyOptRidgeRegression, self).__init__(configs)
        self.cv_params['C'] = 10**np.asarray(range(-4,4),dtype='float64')

    def train(self, data):
        f = self.create_ridge_regression_eval(data, self.C)
        g = self.create_ridge_regression_gradient(data, self.C)
        method = 'BFGS'
        options = {
            'disp': True
        }
        w0 = np.zeros(data.p+1)
        #results2 = optimize.minimize(f,w0,method=method,options=options)
        results = optimize.minimize(f,w0,method=method,jac=g,options=options)
        self.w = results.x[1:]
        self.b = results.x[0]
        pass

    def create_ridge_regression_eval(self, data,C):
        to_use = data.is_labeled & data.is_train
        x = data.x[to_use,:]
        #x = array_functions.add_bias(x)
        y = data.y[to_use]
        f = lambda w: norm(x.dot(w[1:]) + w[0] - y)**2 + C*norm(w[1:])**2
        return f

    def create_ridge_regression_gradient(self,data,C):
        to_use = data.is_labeled & data.is_train
        n = sum(to_use)
        x = data.x[to_use,:]
        #x = array_functions.add_bias(x)
        y = data.y[to_use]
        xx = x.T.dot(x)
        x1 = x.T.dot(np.ones(n))
        xy = x.T.dot(y)
        y1 = y.sum()
        f = lambda w: np.append(
            2*(w[0]*n - y1 + x1.dot(w[1:])),
            2*(xx.dot(w[1:]) + x1*w[0] - xy + C*w[1:])
        )
        return f

class ScipyOptCombinePrediction(method.Method):
    def __init__(self,configs=base_configs.MethodConfigs()):
        super(ScipyOptCombinePrediction, self).__init__(configs)
        self.cv_params['C'] = 10**np.asarray(range(-4,4),dtype='float64')
        self.max_value = 1

    def train(self, data):
        f = self.create_eval(data, self.C)
        g = self.create_gradient(data, self.C)
        method = 'L-BFGS-B'
        options = {
            'disp': False,
            'maxiter': np.inf,
            'maxfun': np.inf
        }
        w0 = np.zeros(data.p+1)
        x = data.x
        a = np.squeeze(data.a)
        b = np.squeeze(data.b)
        args = (x,a,b,self.C,self.max_value)
        results = optimize.minimize(f,w0,method=method,jac=g,options=options,args=args)
        compare_results = False
        if compare_results:
            results2 = optimize.minimize(f,w0,method=method,options=options,args=args)
            err = results.x - results2.x
            print 'Rel Error - w: ' + str(norm(err[1:])/norm(results2.x[1:]))
            print 'Rel Error - bias: ' + str(norm(err[0])/norm(results2.x[0]))
            print 'Rel Error - f(w*): ' + str(norm(results.fun-results2.fun)/norm(results2.fun))
            results = results2

        self.w = results.x[1:]
        self.b = results.x[0]
        pass

    def predict(self, data):
        #sig = scipy.special.expit(data.x.dot(self.w) + self.b)
        sig = ScipyOptCombinePrediction.sigmoid(data.x,self.w,self.b,self.max_value)
        a = np.squeeze(data.a)
        p = np.multiply(sig,a)

        o =  results.Output(data)
        o.fu = p

        if not data.is_regression:
            #y = p.argmax(1)
            y = p
        else:
            y = p
        o.y = y
        return o

    def combine_predictions(self,x,y_source,y_target):
        #s = scipy.special.expit(x.dot(self.w) + self.b)
        s = ScipyOptCombinePrediction.sigmoid(x,self.w,self.b,self.max_value)
        return np.multiply(s,y_source) + np.multiply(1-s,y_target)

    def predict_g(self, x):
        if x.ndim == 1:
            x = np.expand_dims(x,1)
        #g = scipy.special.expit(x.dot(self.w) + self.b)
        g = ScipyOptCombinePrediction.sigmoid(x,self.w,self.b,self.max_value)
        return g

    @staticmethod
    def sigmoid(x, w, b, max_value):
        k = 1
        c = k*(x.dot(w) + b)
        return max_value*scipy.special.expit(c)

    @staticmethod
    def eval(w,x,a,b,C,max_value):
        #w[0] = 0
        #sig = scipy.special.expit(x.dot(w[1:]) + w[0])
        sig = ScipyOptCombinePrediction.sigmoid(x,w[1:],w[0],max_value)
        #sig2 = 1 / (1 + math.exp(-x[0,:].dot(w[1:]) - w[0]))
        #diff = sig[0]-sig2
        #assert math.fabs(diff) < 1e-12
        return norm(np.multiply(sig,a) + b)**2 + C*norm(w[1:])**2

    @staticmethod
    def gradient(w,x,a,b,C,max_value):
        #w[0] = 0
        n = x.shape[0]
        p = x.shape[1]


        #sig = scipy.special.expit(x.dot(w[1:]) + w[0])
        sig = ScipyOptCombinePrediction.sigmoid(x,w[1:],w[0],max_value)
        t = np.multiply(sig,a) + b
        t = np.multiply(t,a)
        ss = np.multiply(sig,1-sig)
        t = np.multiply(t,ss)
        t *= 2
        t *= max_value
        #Regularization component of gradient

        g = x.T.dot(t)
        g = np.insert(g,0,t.sum())
        return g


    def create_eval(self, data,C):
        return ScipyOptCombinePrediction.eval

    def create_gradient(self,data,C):
        return ScipyOptCombinePrediction.gradient

