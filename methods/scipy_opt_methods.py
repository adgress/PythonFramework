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
from loss_functions import loss_function
from timer.timer import tic,toc


#TODO: subclass ScipyOptMethod
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


class ScipyOptMethod(method.Method):
    @staticmethod
    def eval(w,x,a,b,C,max_value):
        return None

    @staticmethod
    def gradient(w,x,a,b,C,max_value):
        return None

    def create_eval(self, data,C):
        return self.__class__.eval

    def create_gradient(self,data,C):
        return self.__class__.gradient


class ScipyOptNonparametricHypothesisTransfer(ScipyOptMethod):
    def __init__(self,configs=base_configs.MethodConfigs()):
        super(ScipyOptNonparametricHypothesisTransfer, self).__init__(configs)
        self.cv_params['C'] = 10**np.asarray(range(-4,4),dtype='float64')
        self.g_nw = method.NadarayaWatsonMethod(configs)
        self.g_nw.configs.target_labels = None
        self.g_nw.configs.source_labels = None
        self.g_nw.configs.cv_loss_function = loss_function.MeanSquaredError()
        self.g_nw.quiet = True
        self.k = 3
        self.metric = configs.metric

    @staticmethod
    def fused_lasso(x, W):
        n = x.shape[0]
        val = 0
        for i in range(n):
            for j in range(i+1,n):
               val += abs(x[i]-x[j])*W[i,j]

        return val

    def train(self, data):
        f = self.create_eval(data, self.C)
        g = self.create_gradient(data, self.C)
        bounds = list((0, None) for i in range(data.n))
        g0 = np.zeros(data.n)
        x = data.x
        y_s = np.squeeze(data.y_s[:,0])
        y_t = np.squeeze(data.y_t[:,0])
        y = data.y
        W = -array_functions.make_laplacian_kNN(data.x,self.k,self.configs.metric)
        W = array_functions.try_toarray(W)
        if not data.is_regression:
            y = array_functions.make_label_matrix(data.y)[:,data.classes].toarray()
            y = y[:,0]
        reg = self.create_reg(data.x)
        reg2 = self.create_reg2(data.x)
        if self.configs.use_fused_lasso:
            method = 'SLSQP'
            max_iter = 1000
            maxfun = 1000
            fused_lasso = ScipyOptNonparametricHypothesisTransfer.fused_lasso
            lasso = lambda x : self.C - fused_lasso(x,W)
            constraints = {
                'type': 'ineq',
                'fun': lasso
            }
            args = (x,y,y_s,y_t,0,reg,self.C2,reg2)
        else:
            method = 'L-BFGS-B'
            max_iter = np.inf
            max_fun = np.inf
            constraints = None
            args = (x,y,y_s,y_t,self.C,reg,self.C2,reg2)

        options = {
            'disp': False,
            'maxiter':max_iter,
            'maxfun': maxfun
        }
        results = optimize.minimize(
            f,
            g0,
            method=method,
            bounds=bounds,
            jac=g,
            options=options,
            constraints=constraints,
            args=args
        )
        compare_results = False
        if compare_results:
            results2 = optimize.minimize(
                f,
                g0,
                method=method,
                bounds=bounds,
                options=options,
                constraints=constraints,
                args=args
            )
            err = results.x - results2.x
            if norm(results2.x) == 0:
                print 'All zeros - using absolute error'
                print 'Abs Error - g: ' + str(norm(err))
            else:
                print 'Rel Error - g: ' + str(norm(err)/norm(results2.x))
            rel_error = norm(results.fun-results2.fun)/norm(results2.fun)
            print 'Rel Error - f(g*): ' + str(rel_error)
            if  rel_error > .001 and norm(results2.x) > 0:
                print 'Big error: C=' + str(self.C) + ' C2=' + str(self.C2)
            results = results2

        '''
        I = data.arg_sort()
        x = (data.x[I,:])
        g = array_functions.vec_to_2d(results.x[I])
        v = np.hstack((x,g))
        print v
        print ''
        '''
        self.g = results.x
        g_data = data_lib.Data()
        g_data.x = data.x
        g_data.y = results.x
        g_data.is_regression = True
        g_data.set_defaults()
        self.g_nw.train_and_test(g_data)

    def predict(self, data):
        fu = self.combine_predictions(data.x,data.y_s,data.y_t)
        o =  results.Output(data)
        o.fu = fu
        o.y = fu
        return o

    def combine_predictions(self,x,y_source,y_target):
        data = data_lib.Data()
        data.x = x
        data.is_regression = True
        g = self.g_nw.predict(data).fu
        a_t = 1 / (1 + g)
        b_s = g / (1 + g)
        if y_source.ndim > 1:
            a_t = array_functions.vec_to_2d(a_t)
            b_s = array_functions.vec_to_2d(b_s)
            fu = a_t*y_target + b_s*y_source
        else:
            fu = np.multiply(a_t, y_target) + np.multiply(b_s, y_source)
        return fu

    def predict_g(self, x):
        data = data_lib.Data()
        data.x = x
        data.is_regression = True
        g = self.g_nw.predict(data).fu
        return g

    def create_reg(self,x):
        L = array_functions.make_laplacian_kNN(x,self.k,self.metric)
        r = lambda g: ScipyOptNonparametricHypothesisTransfer.reg(g,L)
        return r

    def create_reg2(self,x):
        r = lambda g: ScipyOptNonparametricHypothesisTransfer.reg2(g)
        return r

    @staticmethod
    def reg2(g):
        return norm(g)**2, 2*g

    @staticmethod
    def reg(g,L):
        Lg = L.dot(g)
        val = Lg.dot(g)
        return val, 2*Lg

    @staticmethod
    def eval(g,x,y,y_s,y_t,C,reg,C2,reg2):
        err = ScipyOptNonparametricHypothesisTransfer.error(g,y,y_s,y_t)
        val_reg, unused = reg(g)
        val_reg2, unused = reg2(g)
        val = norm(err)**2
        val += (C*val_reg + C2*val_reg2)
        return val

    @staticmethod
    def error(g,y,y_s,y_t):
        denom = 1 / (1+g)
        a_t = denom
        a_s = np.multiply(g,denom)
        err = np.multiply(y_t,a_t)
        err += np.multiply(y_s,a_s)
        err -=  y
        return err

    @staticmethod
    def gradient(g,x,y,y_s,y_t,C,reg,C2,reg2):
        err = ScipyOptNonparametricHypothesisTransfer.error(g,y,y_s,y_t)
        denom = 1 / np.square(1+g)
        a_t = -denom
        a_s = denom
        grad_loss = np.multiply(a_t,y_t)
        grad_loss += np.multiply(a_s,y_s)
        grad_loss = np.multiply(grad_loss,err)
        unused, grad_reg = reg(g)
        unused, grad_reg2 = reg2(g)
        grad = 2*grad_loss
        grad += (C*grad_reg + C2*grad_reg2)
        #grad *= 2
        return grad

    @property
    def prefix(self):
        s = 'NonParaHypTrans'
        if self.configs.use_fused_lasso:
            s += '-l1'
        if self.configs.use_reg2:
            s += '-reg2'
        return s



class ScipyOptCombinePrediction(ScipyOptMethod):
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

    @property
    def prefix(self):
        return 'SigComb'

