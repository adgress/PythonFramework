

#from sets import Set
import cvxpy as cvx
import numpy as np
from sklearn.preprocessing import StandardScaler

from configs.base_configs import MethodConfigs
from data.data import Constraint
from methods.method import Method, SKLRidgeRegression
from results_class.results import Output
from utility import array_functions
from utility import cvx_logistic


class PairwiseConstraint(Constraint):
    def __init__(self, x1, x2):
        super(PairwiseConstraint, self).__init__()
        self.x.append(x1)
        self.x.append(x2)

    def to_cvx(self, w):
        x1 = self.x[0]
        x2 = self.x[1]
        d = (x1 - x2)*w
        return cvx_logistic.logistic(d)

class RelativeRegressionMethod(Method):
    METHOD_ANALYTIC = 1
    METHOD_CVX = 2
    METHOD_RIDGE = 3
    METHOD_RIDGE_SURROGATE = 4
    METHOD_CVX_LOGISTIC = 5
    METHOD_CVX_LOGISTIC_WITH_LOG = 6
    METHOD_CVX_LOGISTIC_WITH_LOG_NEG = 7
    METHOD_CVX_LOGISTIC_WITH_LOG_SCALE = 8
    CVX_METHODS = {
        METHOD_CVX,
        METHOD_CVX_LOGISTIC,
        METHOD_CVX_LOGISTIC_WITH_LOG,
        METHOD_CVX_LOGISTIC_WITH_LOG_NEG,
        METHOD_CVX_LOGISTIC_WITH_LOG_SCALE
    }
    CVX_METHODS_LOGISTIC = {
        METHOD_CVX_LOGISTIC,
        METHOD_CVX_LOGISTIC_WITH_LOG,
        METHOD_CVX_LOGISTIC_WITH_LOG_NEG,
        METHOD_CVX_LOGISTIC_WITH_LOG_SCALE
    }
    CVX_METHODS_LOGISTIC_WITH_LOG = {
        METHOD_CVX_LOGISTIC_WITH_LOG,
        METHOD_CVX_LOGISTIC_WITH_LOG_NEG,
        METHOD_CVX_LOGISTIC_WITH_LOG_SCALE
    }
    METHOD_NAMES = {
        METHOD_ANALYTIC: 'analytic',
        METHOD_CVX: 'cvx',
        METHOD_RIDGE: 'ridge',
        METHOD_RIDGE_SURROGATE: 'ridge-surr',
        METHOD_CVX_LOGISTIC: 'cvx-log',
        METHOD_CVX_LOGISTIC_WITH_LOG: 'cvx-log-with-log',
        METHOD_CVX_LOGISTIC_WITH_LOG_NEG: 'cvx-log-with-log-neg',
        METHOD_CVX_LOGISTIC_WITH_LOG_SCALE: 'cvx-log-with-log-scale'
    }
    def __init__(self,configs=MethodConfigs()):
        super(RelativeRegressionMethod, self).__init__(configs)
        self.can_use_test_error_for_model_selection = True
        self.cv_params['C'] = 10**np.asarray(list(reversed(range(-8,8))),dtype='float64')
        self.cv_params['C2'] = 10**np.asarray(list(reversed(range(-8,8))),dtype='float64')
        self.w = None
        self.b = None
        self.transform = StandardScaler()
        self.add_random_pairwise = True
        self.use_pairwise = configs.use_pairwise
        self.num_pairwise = configs.num_pairwise
        self.use_test_error_for_model_selection = False
        self.no_linear_term = True
        self.neg_log = False

        self.method = RelativeRegressionMethod.METHOD_CVX_LOGISTIC_WITH_LOG

        if not self.use_pairwise:
            self.cv_params['C2'] = np.asarray([0])

    def train(self, data):
        if self.add_random_pairwise:
            data.pairwise_relationships = set()
            I = data.is_train & ~data.is_labeled
            sampled_pairs = array_functions.sample_pairs(I.nonzero()[0], self.num_pairwise)
            for i,j in sampled_pairs:
                pair = (i,j)
                if data.true_y[j] <= data.true_y[i]:
                    pair = (j,i)
                #data.pairwise_relationships.add(pair)
                x1 = data.x[pair[0],:]
                x2 = data.x[pair[1],:]
                data.pairwise_relationships.add(PairwiseConstraint(x1,x2))
        is_labeled_train = data.is_train & data.is_labeled
        labeled_train = data.labeled_training_data()
        x = labeled_train.x
        y = labeled_train.y
        x_orig = x
        x = self.transform.fit_transform(x, y)

        use_ridge = self.method in {
            RelativeRegressionMethod.METHOD_RIDGE,
            RelativeRegressionMethod.METHOD_RIDGE_SURROGATE
        }
        n, p = x.shape
        if use_ridge:
            ridge_reg = SKLRidgeRegression(self.configs)
            ridge_reg.set_params(alpha=self.C)
            ridge_reg.set_params(normalize=False)
            '''
            d = deepcopy(data)
            d.x[is_labeled_train,:] = x
            ridge_reg.train(d)
            '''
            ridge_reg.train(data)
            w_ridge = array_functions.vec_to_2d(ridge_reg.skl_method.coef_)
            b_ridge = ridge_reg.skl_method.intercept_
            self.w = w_ridge
            self.b = b_ridge
            self.ridge_reg = ridge_reg
        elif self.method == RelativeRegressionMethod.METHOD_ANALYTIC:
            x_bias = np.hstack((x,np.ones((n,1))))
            A = np.eye(p+1)
            A[p,p] = 0
            XX = x_bias.T.dot(x_bias)
            v = np.linalg.lstsq(XX + self.C*A,x_bias.T.dot(y))
            w_anal = array_functions.vec_to_2d(v[0][0:p])
            b_anal = v[0][p]
            self.w = w_anal
            self.b = b_anal
        elif self.method in RelativeRegressionMethod.CVX_METHODS:
            w = cvx.Variable(p)
            b = cvx.Variable(1)
            loss = cvx.sum_entries(
                cvx.power(
                    x*w + b - y,
                    2
                )
            )
            reg = cvx.norm(w)**2
            pairwise_reg2 = 0
            assert self.no_linear_term
            for c in data.pairwise_relationships:
                pairwise_reg2 += c.to_cvx(w)
            '''
            for i,j in data.pairwise_relationships:

                #x1 <= x2
                x1 = self.transform.transform(data.x[i,:])
                x2 = self.transform.transform(data.x[j,:])
                if self.method == RelativeRegressionMethod.METHOD_CVX:
                    pairwise_reg += (x1 - x2)*w
                elif self.method in RelativeRegressionMethod.CVX_METHODS_LOGISTIC:
                    a = (x1 - x2)*w
                    if self.method == RelativeRegressionMethod.METHOD_CVX_LOGISTIC:
                        pairwise_reg += self.C2*a
                    elif self.method in RelativeRegressionMethod.CVX_METHODS_LOGISTIC_WITH_LOG:
                        pairwise_reg += self.C2*a
                        if self.C2 == 0:
                            continue
                        a2 = (x1 - x2)*w
                        if self.method != RelativeRegressionMethod.METHOD_CVX_LOGISTIC_WITH_LOG_SCALE:
                            a2 *= self.C2
                        if self.method == RelativeRegressionMethod.METHOD_CVX_LOGISTIC_WITH_LOG_NEG or self.neg_log:
                            a2 = -a2
                        from utility import cvx_logistic
                        a3 = cvx_logistic.logistic(a2)
                        if self.method == RelativeRegressionMethod.METHOD_CVX_LOGISTIC_WITH_LOG_SCALE:
                            a3 *= self.C2
                        pairwise_reg2 += a3
                else:
                    assert False, 'Unknown CVX Method'
            if self.no_linear_term:
                pairwise_reg = 0
            '''
            constraints = []
            obj = cvx.Minimize(loss + self.C*reg + self.C2*pairwise_reg2)
            prob = cvx.Problem(obj,constraints)
            assert prob.is_dcp()
            try:
                prob.solve(verbose=False)
                w_value = w.value
                b_value = b.value
                #print prob.status
                assert w_value is not None and b_value is not None
                #print a.value
                #print b.value
            except:
                #print 'cvx status: ' + str(prob.status)
                k = 0
                w_value = k*np.zeros((p,1))
                b_value = 0
            self.w = w_value
            self.b = b_value
            '''
            obj2 = cvx.Minimize(loss + self.C*reg)
            try:
                prob2 = cvx.Problem(obj2, constraints)
                prob2.solve()
                w2 = w.value
                b2 = b.value
                print 'b error: ' + str(array_functions.relative_error(b_value,b2))
                print 'w error: ' + str(array_functions.relative_error(w_value,w2))
                print 'pairwise_reg value: ' + str(pairwise_reg.value)
            except:
                pass
            '''
        '''
        print 'w rel error: ' + str(array_functions.relative_error(w_value,w_ridge))
        #print 'b rel error: ' + str(array_functions.relative_error(b_value,b_ridge))

        print 'w analytic rel error: ' + str(array_functions.relative_error(w_value,w_anal))
        #print 'b analytic rel error: ' + str(array_functions.relative_error(b_value,b_anal))
        print 'w norm: ' + str(norm(w_value))
        print 'w analytic norm: ' + str(norm(w_anal))
        print 'w ridge norm: ' + str(norm(w_ridge))
        assert self.b is not None
        '''

    def predict(self, data):
        o = Output(data)

        if self.method == RelativeRegressionMethod.METHOD_RIDGE_SURROGATE:
            o = self.ridge_reg.predict(data)
        else:
            x = self.transform.transform(data.x)
            y = x.dot(self.w) + self.b
            o.fu = y
            o.y = y
        return o

    @property
    def prefix(self):
        s = 'RelReg'
        if self.method != RelativeRegressionMethod.METHOD_CVX:
            s += '-' + RelativeRegressionMethod.METHOD_NAMES[self.method]
        if not self.use_pairwise:
            s += '-noPairwiseReg'
        else:
            if self.num_pairwise > 0 and self.add_random_pairwise:
                s += '-numRandPairs=' + str(int(self.num_pairwise))
            if self.no_linear_term:
                s += '-noLinear'
            if self.neg_log:
                s += '-negLog'
        if self.use_test_error_for_model_selection:
            s += '-TEST'
        return s