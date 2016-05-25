

#from sets import Set
import abc
from data.data import Constraint
from utility import cvx_logistic
import cvxpy as cvx
import numpy as np

class CVXConstraint(Constraint):
    def __init__(self):
        super(CVXConstraint, self).__init__()
        self.true_y = []

    def is_pairwise(self):
        return False

    def is_tertiary(self):
        return False

    def cvx_loss_logistic(self, d):
        #return cvx_logistic.logistic(d)
        return cvx.logistic(d)

    @abc.abstractmethod
    def predict(self, f):
        pass

    def is_correct(self, f):
        return self.predict(f)

    @abc.abstractmethod
    def flip(self):
        pass
    '''
    a: lower bound
    b: upper bound
    '''
    def cvx_loss_piecewise(self, d, a, b):
        a_is_none = a is None
        b_is_none = b is None
        assert a_is_none or b_is_none
        assert not (a_is_none and b_is_none), 'Constraint with both upper and lower bounds not implemented yet'
        if not a_is_none:
            v = [0,d]
        elif not b_is_none:
            v = [0,d]
        return cvx.max_elemwise(*v)

class EqualsConstraint(CVXConstraint):
    def __init__(self, x, y):
        super(EqualsConstraint, self).__init__()
        self.x = [x]
        self.y = y

    def to_cvx(self, f):
        return cvx.square(f(self.x[0]) - self.y)

    @staticmethod
    def create_quantize_constraint(data, instance_index, num_quantiles):
        quantiles = data.get_quantiles(num_quantiles)
        x = data.x[instance_index, :]
        y = data.true_y[instance_index]
        #i = np.digitize(y, quantiles)
        i = np.square(quantiles - y).argmin()
        constraint = EqualsConstraint(x, quantiles[i])
        return constraint

class NeighborConstraint(CVXConstraint):
    def __init__(self, x, x_close, x_far):
        super(NeighborConstraint, self).__init__()
        self.x = [x, x_close, x_far]

    def to_cvx(self, f):
        assert False, 'Update this'
        d_close,d_far = self.get_convex_terms(f)
        d = d_close - d_far
        e = cvx.max_elemwise(d,0)
        return e

    def get_convex_terms(self, f):
        d_close = cvx.square(f(self.x[0]) - f(self.x[1]))
        d_far = cvx.square(f(self.x[0]) - f(self.x[2]))
        return d_close,d_far

    def predict(self, f):
        y0 = f(self.x[0])
        return abs(y0-f(self.x[1])) <= abs(y0 - f(self.x[2]))

    def flip(self):
        x = self.x[1]
        self.x[1] = self.x[2]
        self.x[2] = x

    def is_tertiary(self):
        return True

    @staticmethod
    def to_cvx_dccp(constraints, f, logistic=False):
        objective = 0
        n = len(constraints)
        t = cvx.Variable(n)
        t_constraints = [0]*n
        #d_far - d_close
        for i,c in enumerate(constraints):
            d_close,d_far = c.get_convex_terms(f)
            '''
            t_constraints[i] = t[i] == d_close
            assert False, 'Is this correct?'
            objective += cvx.max_elemwise(d_far - t[i], 0)
            '''
            t_constraints[i] = t[i] == d_far
            if logistic:
                objective += cvx.logistic(d_close-t[i])
            else:
                objective += cvx.max_elemwise(d_close-t[i], 0)
        return objective, t, t_constraints

#y1 <= y2
class PairwiseConstraint(CVXConstraint):
    def __init__(self, x1, x2):
        super(PairwiseConstraint, self).__init__()
        self.x.append(x1)
        self.x.append(x2)

    def predict(self, f):
        y0 = f(self.x[0])
        y1 = f(self.x[1])
        return y0 < y1

    def flip(self):
        x = self.x[0]
        self.x[0] = self.x[1]
        self.x[1] = x

    def to_cvx(self, f, scale=1.0):
        x1 = self.x[0]
        x2 = self.x[1]
        d = f(x1) - f(x2)
        return self.cvx_loss_logistic(d/scale)

    def is_pairwise(self):
        return True

class HingePairwiseConstraint(PairwiseConstraint):
    def __init__(self, x1, x2):
        super(HingePairwiseConstraint, self).__init__(x1, x2)

    def to_cvx(self, f):
        x1 = self.x[0]
        x2 = self.x[1]
        d = f(x1) - f(x2)
        return cvx.max_elemwise(d,0)

class SimilarConstraint(PairwiseConstraint):
    def __init__(self, x1, x2):
        super(SimilarConstraint, self).__init__(x1,x2)

    def predict(self, f):
        return False
        x0 = self.x[0]
        x1 = self.x[1]
        return abs(f(x0)-f(x1)) <= self.scale

    def flip(self):
        assert False, 'TODO'

    def to_cvx(self, f, scale=1.0):
        x0 = self.x[0]
        x1 = self.x[1]
        d = f(x0) - f(x1)
        self.scale = scale
        return cvx_logistic.logistic_similar(d, scale)

class BoundConstraint(CVXConstraint):
    BOUND_LOWER = 0
    BOUND_UPPER = 1
    def __init__(self, x, c, bound_type):
        super(BoundConstraint, self).__init__()
        self.x.append(x)
        self.c.append(c)
        self.bound_type = bound_type

    def predict(self, f):
        y0 = f(self.x[0])
        c0 = self.c[0]
        if self.bound_type == BoundConstraint.BOUND_LOWER:
            return y0 >= c0
        else:
            return y0 <= c0

    def flip(self):
        if self.bound_type == BoundConstraint.BOUND_LOWER:
            self.bound_type = BoundConstraint.BOUND_UPPER
        else:
            self.bound_type = BoundConstraint.BOUND_LOWER

    def to_cvx(self, f):
        x = self.x[0]
        c = self.c[0]
        if self.bound_type == BoundConstraint.BOUND_LOWER:
            d = f(x) - c
            a = c
            b = None
        elif self.bound_type == BoundConstraint.BOUND_UPPER:
            d = c - f(x)
            a = None
            b = c
        else:
            assert False
        d *= -1
        return self.cvx_loss_piecewise(d, a, b)

    @staticmethod
    def create_quartile_constraints(data, instance_index):
        quartiles = data.get_quartiles()
        x = data.x[instance_index, :]
        y = data.true_y[instance_index]
        i = np.digitize(y, quartiles)
        if i >= quartiles.size:
            i = quartiles.size-1
        lower = BoundLowerConstraint(x,quartiles[i-1])
        upper = BoundUpperConstraint(x,quartiles[i])
        return (lower,upper)



class BoundLowerConstraint(BoundConstraint):
    def __init__(self, x, c):
        super(BoundLowerConstraint, self).__init__(x,c,BoundConstraint.BOUND_LOWER)

class BoundUpperConstraint(BoundConstraint):
    def __init__(self, x, c):
        super(BoundUpperConstraint, self).__init__(x,c,BoundConstraint.BOUND_UPPER)








