import numpy as np
from numpy.linalg import norm
import cvxpy as cvx
import method
from utility import array_functions
from configs.base_configs import MethodConfigs
from results_class.results import Output
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import linear_model
import logistic_difference_optimize
import scipy.optimize as optimize
from utility.capturing import Capturing

class optimize_data(object):
    def __init__(self, x, y, reg_ridge, reg_a, reg_mixed):
        self.x = x
        self.y = y
        self.reg_ridge = reg_ridge
        self.reg_a = reg_a
        self.reg_mixed = reg_mixed

    def get_xy(self):
        return self.x, self.y

    def get_reg(self):
        return self.reg_ridge, self.reg_a, self.reg_mixed

class MixedFeatureGuidanceMethod(method.Method):
    METHOD_NO_RELATIVE = 1
    METHOD_RIDGE = 2
    METHOD_ORACLE_WEIGHTS = 3
    METHOD_ORACLE_SPARSITY = 4
    METHODS_NO_C2 = {
        METHOD_RIDGE, METHOD_ORACLE_WEIGHTS
    }
    METHODS_NO_C3 = {
        METHOD_NO_RELATIVE, METHOD_RIDGE, METHOD_ORACLE_WEIGHTS
    }
    def __init__(self,configs=MethodConfigs()):
        super(MixedFeatureGuidanceMethod, self).__init__(configs)
        self.cv_params['C'] = self.create_cv_params(-5, 5)
        self.cv_params['C2'] = self.create_cv_params(-5, 5)
        self.cv_params['C3'] = self.create_cv_params(-5, 5)
        self.transform = StandardScaler()
        self.method = MixedFeatureGuidanceMethod.METHOD_NO_RELATIVE
        #self.method = MixedFeatureGuidanceMethod.METHOD_RIDGE
        #self.method = MixedFeatureGuidanceMethod.METHOD_ORACLE_WEIGHTS
        #self.method = MixedFeatureGuidanceMethod.METHOD_ORACLE_SPARSITY
        self.can_use_test_error_for_model_selection = True
        self.use_test_error_for_model_selection = configs.use_test_error_for_model_selection
        if self.method in MixedFeatureGuidanceMethod.METHODS_NO_C3:
            self.C3 = 0
            del self.cv_params['C3']

        if self.method in MixedFeatureGuidanceMethod.METHODS_NO_C2:
            self.C2 = 0
            del self.cv_params['C2']

    def train(self, data):
        assert data.is_regression
        self.is_classifier = not data.is_regression
        return self.solve(data)

    @staticmethod
    def solve_w(a, x, y, C):
        try:
            assert False
            w = np.linalg.lstsq(x.T.dot(x) + C * np.diag(a), x.T.dot(y))[0]
        except Exception as e:
            w = np.zeros(x.shape[1])
        return w

    @staticmethod
    #def eval(a, x, y, C, C2, C3):
    def eval(data, a):
        x, y = data.get_xy()
        C, C2, C3 = data.get_reg()
        n = x.shape[0]
        #p = x.shape[1]
        t = StandardScaler()
        #D_a = np.diag(a)
        loss = 0
        for i in range(n):
            I = array_functions.true(n)
            I[i] = False
            xi = x[i, :]
            yi = y[i]
            xmi = t.fit_transform(x[I, :])
            ymi = y[I]
            bi = ymi.mean()
            #w = np.linalg.lstsq(xmi.T.dot(xmi) + C*D_a, xmi.T.dot(ymi))[0]
            w = MixedFeatureGuidanceMethod.solve_w(a, xmi, ymi, C)
            loss += (xi.T.dot(w) + bi - yi)**2
            '''
            ridge = linear_model.Ridge(C, normalize=False)
            ridge.fit(xmi, ymi)
            b_ridge = ridge.intercept_
            w_ridge = ridge.coef_
            rel_err = array_functions.relative_error(w_ridge, w)
            '''
            pass
        reg = C2*norm(a)**2
        return loss + reg


    def create_grad(self, x, y, C, C2, C3):
        pass

    def solve(self, data):
        is_labeled_train = data.is_train & data.is_labeled
        x = data.x[is_labeled_train, :]
        y = data.y[is_labeled_train]
        n = x.shape[0]
        p = x.shape[1]

        C = self.C
        C2 = self.C2
        C3 = self.C3
        #C = 1
        #C2 = .001
        #eval = self.create_eval(x, y, C, C2, C3)
        self.a = np.ones(p)
        irrelevant_features = array_functions.false(p)
        big_float = 1e16
        irrelevant_features[4:] = True

        if self.method == MixedFeatureGuidanceMethod.METHOD_ORACLE_WEIGHTS:
            self.a[irrelevant_features] = big_float
        elif self.C2 != 0 or self.C3 != 0:
            opt_data = optimize_data(x, y, C, C2, C3)
            eval_func = lambda a: MixedFeatureGuidanceMethod.eval(opt_data, a)
            #MixedFeatureGuidanceMethod.eval(opt_data, np.ones(p))
            a0 = np.ones(p)
            '''
            constraints = [{
                'type': 'ineq',
                'fun': lambda a: a
            }]
            '''
            bounds = [(0, None)]*p
            if self.method == MixedFeatureGuidanceMethod.METHOD_ORACLE_SPARSITY:
                bounds[irrelevant_features] = (big_float, big_float)
            constraints = None

            options = {}
            options['maxiter'] = 1000
            options['disp'] = False
            #with Capturing() as output:
            results = optimize.minimize(
                eval_func,
                a0,
                method=self.configs.scipy_opt_method,
                jac=None,
                options=options,
                bounds = bounds,
                constraints=constraints
            )
            self.a = results.x
        self.b = y.mean()
        x = self.transform.fit_transform(x)
        self.w = MixedFeatureGuidanceMethod.solve_w(self.a, x, y, C)
        pass



    def predict(self, data):
        o = Output(data)
        #W = pairwise.rbf_kernel(data.x,self.x,self.sigma)

        x = self.transform.transform(data.x)
        o.y = x.dot(self.w) + self.b
        return o


    @property
    def prefix(self):
        s = 'Mixed-feats'
        if self.method == MixedFeatureGuidanceMethod.METHOD_RIDGE:
            s += '_method=Ridge'
        elif self.method == MixedFeatureGuidanceMethod.METHOD_NO_RELATIVE:
            s += '_method=NoRel'
        elif self.method == MixedFeatureGuidanceMethod.METHOD_ORACLE_WEIGHTS:
            s += '_method=OracleWeights'
        elif self.method == MixedFeatureGuidanceMethod.METHOD_ORACLE_SPARSITY:
            s += '_method=OracleSparsity'
        if self.use_test_error_for_model_selection:
            s += '-TEST'
        return s