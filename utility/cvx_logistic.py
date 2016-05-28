#Note: My version of cvxpy was missing this, so I just copied it from github

"""
Copyright 2013 Steven Diamond
This file is part of CVXPY.
CVXPY is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
CVXPY is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with CVXPY.  If not, see <http://www.gnu.org/licenses/>.
"""

import cvxpy.utilities as u
import cvxpy.lin_ops.lin_utils as lu
from cvxpy.atoms.elementwise.elementwise import Elementwise
from cvxpy.atoms.elementwise.exp import exp
from cvxpy.atoms.elementwise.log import log
import cvxpy as cvx
import numpy as np
from cvxpy.expressions.constants.parameter import Parameter
from cvxpy.atoms.atom import Atom

class logistic_difference(Atom):
    def __init__(self, x):
        super(logistic_difference, self).__init__(*x)

    def sign_from_args(self):
        """By default, the sign is the most general of all the argument signs.
        """
        return (True, False)

    def size_from_args(self):
        """Returns sign (is positive, is negative) of the expression.
        """
        #assert False, 'Not Implemented'
        #return True
        return (1,1)

    def is_atom_convex(self):
        """Is the atom convex?
        """
        return True

    def is_atom_concave(self):
        """Is the atom concave?
        """
        return False

    def is_incr(self, idx):
        """Is the composition non-decreasing in argument idx?
        """
        assert False, 'is this true?'
        return False

    def is_decr(self, idx):
        """Is the composition non-increasing in argument idx?
        """
        assert False, 'is this true?'
        return True

    @Elementwise.numpy_numeric
    def numeric(self, values):
        """Evaluates e^x elementwise, adds 1, and takes the log.
        """
        assert False, 'Not Implemented'
        assert values.size == 1
        a = self.s
        b = values[0]
        t = np.exp(a-b) + np.exp(-a-b) + np.exp(-2*b)
        return np.log(1 + t)



    def _grad(self, values):
        """Gives the (sub/super)gradient of the atom w.r.t. each argument.

        Matrix expressions are vectorized, so the gradient is a matrix.

        Args:
            values: A list of numeric values for the arguments.

        Returns:
            A list of SciPy CSC sparse matrices or None.
        """
        assert False, 'Not Implemented'

    @staticmethod
    def graph_implementation(arg_objs, size, data=None):
        """Reduces the atom to an affine expression and list of constraints.

        Parameters
        ----------
        arg_objs : list
            LinExpr for each argument.
        size : tuple
            The size of the resulting expression.
        data :
            Additional data required by the atom.

        Returns
        -------
        tuple
            (LinOp for objective, list of constraints)
        """
        #x = arg_objs[0]
        t = lu.create_var(size)

        # log(1 + exp(x)) <= t <=> exp(-t) + exp(x - t) <= 1
        '''
        obj0, constr0 = exp.graph_implementation([lu.neg_expr(t)], size)
        obj1, constr1 = exp.graph_implementation([lu.sub_expr(x, t)], size)
        lhs = lu.sum_expr([obj0, obj1])
        ones = lu.create_const(np.mat(np.ones(size)), size)
        constr = constr0 + constr1 + [lu.create_leq(lhs, ones)]
        '''

        a = arg_objs[0]
        b = arg_objs[1]

        e_a, e_a_cons = exp.graph_implementation([lu.neg_expr(a)], (1,1))
        e_b, e_b_cons = exp.graph_implementation([lu.neg_expr(b)], (1,1))
        obj0, constr0 = log.graph_implementation(
            [lu.sub_expr(e_b,e_a)],
            (1,1)
        )
        lhs = obj0
        constr = constr0 + e_a_cons + e_b_cons + [lu.create_leq(lhs, t)] + [lu.create_leq(e_a, e_b)]

        return (t, constr)

class logistic_similar(cvx.logistic):
    """:math:`\log(1 + e^{x})`
    This is a special case of log(sum(exp)) that is evaluates to a vector rather
    than to a scalar which is useful for logistic regression.
    """
    def __init__(self, x, s):
        super(logistic_similar, self).__init__(x)
        assert not hasattr(self, 's')
        self.s = s

    def get_data(self):
        """Returns the parameter M.
        """
        return [self.s]

    @Elementwise.numpy_numeric
    def numeric(self, values):
        """Evaluates e^x elementwise, adds 1, and takes the log.
        """
        assert values.size == 1
        a = self.s
        b = values[0]
        t = np.exp(a-b) + np.exp(-a-b) + np.exp(-2*b)
        return np.log(1 + t)

    def is_incr(self, idx):
        """Is the composition non-decreasing in argument idx?
        """
        return False

    def is_decr(self, idx):
        """Is the composition non-increasing in argument idx?
        """
        return True

    def _grad(self, values):
        """Gives the (sub/super)gradient of the atom w.r.t. each argument.

        Matrix expressions are vectorized, so the gradient is a matrix.

        Args:
            values: A list of numeric values for the arguments.

        Returns:
            A list of SciPy CSC sparse matrices or None.
        """
        assert False, 'Not Implemented'
        rows = self.args[0].size[0]*self.args[0].size[1]
        cols = self.size[0]*self.size[1]
        exp_val = np.exp(values[0])
        grad_vals = exp_val/(1 + exp_val)
        return [logistic.elemwise_grad_to_diag(grad_vals, rows, cols)]



    @staticmethod
    def graph_implementation(arg_objs, size, data=None):
        """Reduces the atom to an affine expression and list of constraints.

        Parameters
        ----------
        arg_objs : list
            LinExpr for each argument.
        size : tuple
            The size of the resulting expression.
        data :
            Additional data required by the atom.

        Returns
        -------
        tuple
            (LinOp for objective, list of constraints)
        """
        x = arg_objs[0]
        t = lu.create_var(size)

        # log(1 + exp(x)) <= t <=> exp(-t) + exp(x - t) <= 1
        '''
        obj0, constr0 = exp.graph_implementation([lu.neg_expr(t)], size)
        obj1, constr1 = exp.graph_implementation([lu.sub_expr(x, t)], size)
        lhs = lu.sum_expr([obj0, obj1])
        ones = lu.create_const(np.mat(np.ones(size)), size)
        constr = constr0 + constr1 + [lu.create_leq(lhs, ones)]
        '''
        s = data[0]
        if isinstance(s, Parameter):
            s = lu.create_param(s, (1, 1))
        else: # M is constant.
            s = lu.create_const(s, (1, 1))

        obj0, constr0 = exp.graph_implementation([lu.neg_expr(t)], size)
        obj1, constr1 = exp.graph_implementation([lu.sub_expr(s, lu.sum_expr([t, x]))], size)
        obj2, constr2 = exp.graph_implementation([lu.sub_expr(lu.neg_expr(s), lu.sum_expr([t, x]))], size)
        obj3, constr3 = exp.graph_implementation([lu.sub_expr(lu.neg_expr(t), lu.mul_expr(2, x, size))], size)

        lhs = lu.sum_expr([obj0, obj1, obj2, obj3])
        ones = lu.create_const(np.mat(np.ones(size)), size)
        constr = constr0 + constr1 + constr2 + constr3 + [lu.create_leq(lhs, ones)]


        return (t, constr)

        #return NotImplemented
        #assert False, 'Not Implemented'


if __name__ == '__main__':
    import cvxpy as cvx
    a = cvx.Variable()
    x1 = 1
    y1= 3
    x2 = 2
    y2 = 6
    c = logistic_difference([a*x2, a*x2])
    #c = 0
    obj = cvx.Minimize(cvx.norm2(a*x1 - y1) + c)
    prob = cvx.Problem(obj, [])
    ret = prob.solve(solver=cvx.CVXOPT, verbose=True)
    pass
















