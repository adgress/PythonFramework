

#from sets import Set

from data.data import Constraint
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
