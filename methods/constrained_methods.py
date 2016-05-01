

#from sets import Set

from data.data import Constraint
from utility import cvx_logistic
import cvxpy as cvx
import numpy as np

class CVXConstraint(Constraint):
    def __init__(self):
        super(CVXConstraint, self).__init__()

    def is_pairwise(self):
        return False

    def is_tertiary(self):
        return False

    def cvx_loss_logistic(self, d):
        #return cvx_logistic.logistic(d)
        return cvx.logistic(d)
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


class NeighborConstraint(CVXConstraint):
    def __init__(self, x, x_close, x_far):
        super(NeighborConstraint, self).__init__()
        self.x += [x, x_close, x_far]

    def to_cvx(self, f):
        d_close,d_far = self.get_convex_terms(f)
        d = - (d_close - d_far)
        e = cvx.max_elemwise(d,0)
        return e

    def get_convex_terms(self, f):
        d_close = cvx.abs(f(self.x[0]) - f(self.x[1]))
        d_far = cvx.abs(f(self.x[0]) - f(self.x[2]))
        return d_close,d_far

    def is_tertiary(self):
        return True

    @staticmethod
    def to_cvx_dccp(constraints, f):
        objective = 0
        n = len(constraints)
        t = cvx.Variable(n)
        t_constraints = [0]*n
        #d_far - d_close
        for i,c in enumerate(constraints):
            d_close,d_far = c.get_convex_terms(f)
            t_constraints[i] = t[i] == d_close
            #assert False, 'Is this correct?'
            #objective += cvx.max_elemwise(d_far - t[i], 0)
            objective += cvx.max_elemwise(t[i]-d_far, 0)
        return objective, t, t_constraints


class PairwiseConstraint(CVXConstraint):
    def __init__(self, x1, x2):
        super(PairwiseConstraint, self).__init__()
        self.x.append(x1)
        self.x.append(x2)

    def to_cvx(self, f):
        x1 = self.x[0]
        x2 = self.x[1]
        d = f(x1) - f(x2)
        return self.cvx_loss_logistic(d)

    def is_pairwise(self):
        return True

class BoundConstraint(CVXConstraint):
    BOUND_LOWER = 0
    BOUND_UPPER = 1
    def __init__(self, x, c, bound_type):
        super(BoundConstraint, self).__init__()
        self.x.append(x)
        self.c.append(c)
        self.bound_type = bound_type

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

class BoundLowerConstraint(BoundConstraint):
    def __init__(self, x, c):
        super(BoundLowerConstraint, self).__init__(x,c,BoundConstraint.BOUND_LOWER)

class BoundUpperConstraint(BoundConstraint):
    def __init__(self, x, c):
        super(BoundUpperConstraint, self).__init__(x,c,BoundConstraint.BOUND_UPPER)








