import numpy as np
from numpy.linalg import norm
import scipy.optimize as optimize
from scipy.special import expit as sigmoid


#from http://stackoverflow.com/questions/4474395/staticmethod-and-abc-abstractmethod-will-it-blend
class abstractstatic(staticmethod):
    __slots__ = ()
    def __init__(self, function):
        super(abstractstatic, self).__init__(function)
        function.__isabstractmethod__ = True
    __isabstractmethod__ = True

def pack_linear(w,b):
    return np.append(w,b)

def unpack_linear(v):
    return v[0:v.size-1], v[v.size-1]

def eval_reg_l2(w):
    return norm(w)**2

def grad_reg_l2(w):
    return 2*w

def loss_l2(y1, y2):
    return norm(y1-y2)**2

def grad_linear_loss_l2(x, y, v):
    w,b = unpack_linear(v)
    n = y.size
    grad_w = 2*(x.T.dot(x).dot(w) - x.T.dot(y) + 2*x.T.sum()*b)
    grad_b = 2*(x.dot(w).sum() - 2*y.sum() + 2*n*b)
    return pack_linear(grad_w, grad_b)

def apply_linear(x, w, b=None):
    if b is None:
        w,b = unpack_linear(w)
    return x.dot(w) + b

def eval_linear_loss_l2(x, y, v):
    w, b = unpack_linear(v)
    return loss_l2(apply_linear(x, w, b), y)


class optimize_data(object):
    def __init__(self, x, y, reg, reg_mixed):
        self.x = x
        self.y = y
        self.reg = reg
        self.reg_mixed = reg_mixed

    def get_xy(self):
        return self.x, self.y

    def get_reg(self):
        return self.reg, self.reg_mixed


class logistic_optimize(object):
    @abstractstatic
    def eval_mixed_guidance(data, v):
        pass

    @abstractstatic
    def grad_mixed_guidance(data, v):
        pass

    @abstractstatic
    def _grad_num_mixed_guidance(data, v):
        pass

    @classmethod
    def eval(cls, data, v):
        eval_loss = cls.eval_loss(data, v)
        eval_reg = cls.eval_reg(data, v)
        eval_mixed = cls.eval_mixed_guidance(data, v)
        reg, reg_mixed = data.get_reg()
        v = eval_loss + reg*eval_reg
        if reg_mixed > 0:
            v += eval_mixed*reg_mixed

    @staticmethod
    def eval_loss(data, v):
        x, y = data.get_xy()
        return eval_linear_loss_l2(x, y, v)

    @staticmethod
    def grad_loss(data, v):
        x, y = data.get_xy()
        return grad_linear_loss_l2(x, y, v)

    @staticmethod
    def eval_reg(data, v):
        w, b = unpack_linear(v)
        return eval_reg_l2(w)

    @staticmethod
    def grad_reg(data, v):
        w, b = unpack_linear(v)
        g = grad_reg_l2(w)
        return np.append(g, 0)

    @classmethod
    def grad(cls, data, v):
        grad_loss = cls.grad_loss(data, v)
        grad_mixed = cls.grad_mixed_guidance(data, v)
        reg, reg_mixed = data.get_reg()
        grad_reg = cls.grad_reg(data, v)
        grad_reg *= reg

        I = np.isinf(reg_mixed) | np.isnan(reg_mixed)
        if I.any():
            print 'inf or nan!'
            reg_mixed[I] = 0

        val = grad_loss + grad_reg
        if reg_mixed != 0:
            val += reg_mixed * grad_mixed

        return val

    @classmethod
    def create_eval(cls, data):
        return lambda v: cls.eval(data, v)

    @classmethod
    def create_grad(cls, data):
        return lambda v: cls.grad(data, v)

class logistic_similar(logistic_optimize):
    @staticmethod
    def eval_mixed_guidance(data, v):
        p1, p2 = data.pairs
        yi = apply_linear(p1, v)
        yj = apply_linear(p2, v)
        return sigmoid(yi-yj).sum()

    @staticmethod
    def grad_mixed_guidance(data, v):
        pass

    @staticmethod
    def _grad_num_mixed_guidance(data, v):
        pass

class logistic_neighbor(object):

    @staticmethod
    def eval_mixed_guidance(x, x_low, x_high, v):
        w, b = unpack_linear(v)
        y = apply_linear(x, w, b)
        y_low = apply_linear(x_low, w, b)
        y_high = apply_linear(x_high, w, b)

        sig1 = sigmoid(y_high-y_low)
        sig2 = sigmoid(2*y - y_high - y_low)
        diff = sig1 - sig2
        vals2 = -np.log(sig1-sig2)
        I = np.isinf(vals2) | np.isnan(vals2)
        if I.any():
            print 'eval_linear_neighbor_logistic: inf = ' + str(I.mean())
            vals2[I] = 1e6
        val2 = vals2.sum()
        #assert norm(val - val2)/norm(val) < 1e-6
        return val2

    @staticmethod
    def grad_mixed_guidance(x, x_low, x_high, v):
        w, b = unpack_linear(v)
        y = apply_linear(x, w, b)
        y_low = apply_linear(x_low, w, b)
        y_high = apply_linear(x_high, w, b)

        sig1 = sigmoid(y_high-y_low)
        sig2 = sigmoid(2*y - y_high - y_low)
        denom = sig1 - sig2
        val = np.zeros(v.shape)
        for i in range(x.shape[0]):
            num1, x1, num2, x2 = logistic_neighbor._grad_num_mixed_guidance(x[i, :], x_low[i, :], x_high[i, :], w, b)
            val[0:-1] += (num1*x1 - num2*x2) / denom[i]
            val[-1] += (num1 - num2) / denom[i]
        #assert not np.isnan(val).any()
        #Why isn't this necessary?
        #val *= -1
        I = np.isnan(val) | np.isinf(val)
        if I.any():
            print 'grad_linear_neighbor_logistic: nan!'
            val[I] = 0
        return val

    @staticmethod
    def _grad_num_mixed_guidance(x, x_low, x_high, w, b=None):
        if b is None:
            w,b = unpack_linear(w)


        y = apply_linear(x, w, b)
        y_low = apply_linear(x_low, w, b)
        y_high = apply_linear(x_high, w, b)

        a1 = y_high - y_low
        a2 = 2*y-y_low-y_high

        t1 = np.exp(-a1)*((1 + np.exp(-a1))**-2)
        t2 = np.exp(-a2)*((1 + np.exp(-a2))**-2)

        x1 = x_low - x_high
        x2 = x_high + x_low - 2*x

        return t1,x2,t2,x2

    @staticmethod
    def eval(x, y, x_neighbor, x_low, x_high, reg_w, reg_neighbor, v):
        w, b = unpack_linear(v)
        y_pred = apply_linear(x, w, b)
        loss = eval_linear_loss_l2(x, y, v)
        loss_neighbor = logistic_neighbor.eval_mixed_guidance(x_neighbor, x_low, x_high, v)
        loss_reg = eval_reg_l2(w)

        val = loss + reg_w*loss_reg
        if reg_neighbor != 0:
            val += reg_neighbor*loss_neighbor
        return val

    @staticmethod
    def grad(x, y, x_neighbor, x_low, x_high, reg_w, reg_neighbor, v):
        w, b = unpack_linear(v)
        grad_loss = grad_linear_loss_l2(x, y, v)
        grad_neighbor = logistic_neighbor.grad_mixed_guidance(x_neighbor, x_low, x_high, v)
        grad_reg = grad_reg_l2(w)
        grad_reg *= reg_w
        grad_reg = np.append(grad_reg, 0)

        val = grad_loss + grad_reg
        if reg_neighbor != 0:
            val += reg_neighbor * grad_neighbor
        return val

    @staticmethod
    def create_eval(x, y, x_neighbor, x_low, x_high, reg_w, reg_neighbor):
        return lambda v: logistic_neighbor.eval(x, y, x_neighbor, x_low, x_high, reg_w, reg_neighbor, v)

    @staticmethod
    def create_grad(x, y, x_neighbor, x_low, x_high, reg_w, reg_neighbor):
        return lambda v: logistic_neighbor.grad(x, y, x_neighbor, x_low, x_high, reg_w, reg_neighbor, v)

    @staticmethod
    def constraint_neighbor(v, x_low, x_high):
        w,b = unpack_linear(v)
        y_low = apply_linear(x_low,w,b)
        y_high = apply_linear(x_high,w,b)
        return y_high - y_low

    @staticmethod
    def create_constraint_neighbor(x_low, x_high):
        return lambda v: logistic_neighbor.constraint_neighbor(v, x_low, x_high)

class logistic_bound:
    @staticmethod
    def eval_mixed_guidance(x, bounds, v):
        w, b = unpack_linear(v)
        y = apply_linear(x, w, b)
        assert y.size == bounds.shape[0]
        c1 = bounds[:, 0]
        c2 = bounds[:, 1]
        loss_num = np.log(np.exp(y - c1) - np.exp(y - c2))
        loss_denom = np.log(1 + np.exp(y-c1) + np.exp(y-c2) + np.exp(2*y - c1 - c2))
        vals = (-loss_num + loss_denom)
        val = vals.sum()

        sig1 = sigmoid(c2-y)
        sig2 = sigmoid(c1-y)
        diff = sig1 - sig2
        vals2 = -np.log(sig1-sig2)
        val2 = vals2.sum()
        #assert norm(val - val2)/norm(val) < 1e-6
        return val2

    @staticmethod
    def _grad_num_mixed_guidance(x, c, w, b=None):
        if b is None:
            w,b = unpack_linear(w)
        a = c - apply_linear(x, w, b)
        t = 1 + np.exp(-a)
        t = t ** -2
        t2 = np.exp(-a)
        return t*t2, x

    @staticmethod
    def grad_mixed_guidance(x, bounds, v):
        w, b = unpack_linear(v)
        y = apply_linear(x, w, b)
        assert y.size == bounds.shape[0]
        c1 = bounds[:, 0]
        c2 = bounds[:, 1]

        sig1 = sigmoid(c2-y)
        sig2 = sigmoid(c1-y)
        denom = sig1 - sig2
        val = np.zeros(v.shape)
        for i in range(x.shape[0]):
            num1, x1 = logistic_bound._grad_num_mixed_guidance(x[i, :], c2[i], w, b)
            num2, x2 = logistic_bound._grad_num_mixed_guidance(x[i, :], c1[i], w, b)
            val[0:-1] += (num1*x1 - num2*x2) / denom[i]
            val[-1] += (num1 - num2) / denom[i]
        #assert not np.isnan(val).any()
        #Why isn't this necessary?
        #val *= -1
        if np.isnan(val).any():
            print 'grad_linear_bound_logistic: nan!'
            val[np.isnan(val)] = 0
        return val

    @staticmethod
    def eval(x, y, x_bound, bounds, reg_w, reg_bound, v):
        w, b = unpack_linear(v)
        y_pred = apply_linear(x, w, b)
        loss = eval_linear_loss_l2(x, y, v)
        loss_bound = logistic_bound.eval_mixed_guidance(x_bound, bounds, v)
        loss_reg = eval_reg_l2(w)

        val = loss + reg_w*loss_reg
        if reg_bound != 0:
            val += reg_bound * loss_bound
        return val

    @staticmethod
    def grad(x, y, x_bound, bounds, reg_w, reg_bound, v):
        w, b = unpack_linear(v)
        grad_loss = grad_linear_loss_l2(x, y, v)
        grad_bound = logistic_bound.grad_mixed_guidance(x_bound, bounds, v)
        grad_reg = grad_reg_l2(w)
        grad_reg *= reg_w
        grad_reg = np.append(grad_reg, 0)

        val = grad_loss + grad_reg
        if reg_bound != 0:
            val += reg_bound * grad_bound
        return val

    @staticmethod
    def create_eval(x, y, x_bound, bounds, reg_w, reg_bound):
        return lambda v: logistic_bound.eval(x, y, x_bound, bounds, reg_w, reg_bound, v)

    @staticmethod
    def create_grad(x, y, x_bound, bounds, reg_w, reg_bound):
        return lambda v: logistic_bound.grad(x, y, x_bound, bounds, reg_w, reg_bound, v)

    @staticmethod
    def constraint_bound(v, x, upper_bound):
        w,b = unpack_linear(v)
        y = apply_linear(x, w, b)
        return upper_bound - y

    @staticmethod
    def create_constraint_bound(x, upper_bound):
        return lambda v: logistic_bound.constraint_bound(v, x, upper_bound)

