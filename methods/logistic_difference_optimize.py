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
    grad_w = 2*(x.T.dot(x).dot(w) - x.T.dot(y) + 2*x.T.sum(1)*b)
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
        reg, reg_mixed = data.get_reg()
        val = eval_loss + reg*eval_reg
        if reg_mixed > 0:
            eval_mixed = cls.eval_mixed_guidance(data, v)
            val += eval_mixed*reg_mixed
        return val

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
        reg, reg_mixed = data.get_reg()
        grad_reg = cls.grad_reg(data, v)
        grad_reg *= reg

        I = np.isinf(reg_mixed) | np.isnan(reg_mixed)
        if I.any():
            print 'inf or nan!'
            reg_mixed[I] = 0

        val = grad_loss + grad_reg
        if reg_mixed != 0:
            grad_mixed = cls.grad_mixed_guidance(data, v)
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
        x1 = data.x1
        x2 = data.x2
        s = data.s
        y1 = apply_linear(x1, v)
        y2 = apply_linear(x2, v)
        d = y2 - y1
        denom = np.log(1 + np.exp(s+d) + np.exp(d-s) + np.exp(2*d))

        vals = d - denom + np.log(np.exp(s) - np.exp(-s))
        return -vals.sum()


    @staticmethod
    def grad_mixed_guidance(data, v):
        x1 = data.x1
        x2 = data.x2
        s = data.s
        y1 = apply_linear(x1, v)
        y2 = apply_linear(x2, v)
        d = y2 - y1
        a = np.exp(s+d) + np.exp(d-s) + np.exp(2*d)
        a2 = np.exp(s+d) + np.exp(d-s) + 2*np.exp(2*d)
        n = x1.shape[0]
        g = np.zeros(v.size)

        sig1 = sigmoid(s - d)
        sig2 = sigmoid(-s - d)

        for i in range(n):
            dx = x2[i,:] - x1[i,:]

            ai = a[i]
            a2i = a2[i]
            v = a2i/(1+ai)
            if np.isinf(ai):
                v = 1
            t = 1 - v

            g[0:-1] += t*dx
            g[-1] += t
        g[-1] = 0
        return -g
        #return g

class logistic_pairwise(logistic_optimize):
    @staticmethod
    def eval_mixed_guidance(data, v):
        x_low = data.x_low
        x_high = data.x_high
        yj = apply_linear(x_low, v)
        yi = apply_linear(x_high, v)
        d = yi - yj
        '''
        v = sigmoid(yi-yj)
        v_log = -np.log(v)
        return v_log.sum()
        '''
        vals = np.log(1 + np.exp(-d))
        I = np.isinf(vals)
        if I.any():
            #print 'logistic_pairwise eval: inf! ' + str(I.mean())
            #vals[np.isinf(vals)] = 1e16
            pass
        return vals.sum()

    @staticmethod
    def grad_mixed_guidance(data, v):
        x_low = data.x_low
        x_high = data.x_high
        n = x_low.shape[0]
        d = apply_linear(x_high, v) - apply_linear(x_low, v)
        a = np.exp(-d)
        sig = sigmoid(d)
        g = np.zeros(v.size)
        for i in range(n):
            dx = x_high[i,:] - x_low[i,:]
            #t = a[i]
            #t *= ((1+a[i])**-2)
            #t *= ((1+a[i])**-1)
            t = 1-sig[i]
            g[0:-1] += t*dx
            g[-1] += t
        g *= -1
        g[-1] *= 0
        return g

eps = 1e-2

class logistic_neighbor(logistic_optimize):

    @staticmethod
    def eval_mixed_guidance(data, v):
        x = data.x_neighbor
        x_low = data.x_low
        x_high = data.x_high
        w, b = unpack_linear(v)
        y = apply_linear(x, w, b)
        y_low = apply_linear(x_low, w, b)
        y_high = apply_linear(x_high, w, b)
        '''
        if (y_low + eps >= y_high).any() or (y + eps >= y_high).any():
            return np.inf
        '''
        sig1 = sigmoid(y_high-y_low)
        sig2 = sigmoid(2*y - y_high - y_low)
        diff = sig1 - sig2
        #assert (np.sign(diff) > 0).all()
        vals2 = -np.log(sig1-sig2 + eps)
        I = np.isnan(vals2)
        if I.any():
            #print 'eval_linear_neighbor_logistic: inf = ' + str(I.mean())
            return np.inf
        val2 = vals2.sum()
        #assert norm(val - val2)/norm(val) < 1e-6
        return val2

    @staticmethod
    def grad_mixed_guidance(data, v):
        x = data.x_neighbor
        x_low = data.x_low
        x_high = data.x_high
        w, b = unpack_linear(v)
        y = apply_linear(x, w, b)
        y_low = apply_linear(x_low, w, b)
        y_high = apply_linear(x_high, w, b)

        sig1 = sigmoid(y_high-y_low)
        sig2 = sigmoid(2*y - y_high - y_low)
        denom = sig1 - sig2 + eps
        val = np.zeros(v.shape)
        for i in range(x.shape[0]):
            #num1, x1, num2, x2 = logistic_neighbor._grad_num_mixed_guidance(x[i, :], x_low[i, :], x_high[i, :], w, b)
            #val[0:-1] += (num1*x1 - num2*x2) / denom[i]
            #val[-1] += (num1 - num2) / denom[i]

            x1 = x_high[i,:] - x_low[i,:]
            x2 = 2*x[i,:] - x_low[i,:] - x_high[i,:]
            num1 = sig1[i]*(1-sig1[i])*x1
            num2 = sig2[i]*(1-sig2[i])*x2
            val[0:-1] += (num1-num2)*(1/denom[i])
            #val[-1] += (num1-num2)*(1/denom[i])

        #assert not np.isnan(val).any()
        #Why isn't this necessary?
        #val *= -1
        val[-1] = 0
        val *= -1
        I = np.isnan(val) | np.isinf(val)
        if I.any():
            #print 'grad_linear_neighbor_logistic: nan!'
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
    def constraint_neighbor(v, x_low, x_high):
        w,b = unpack_linear(v)
        y_low = apply_linear(x_low,w,b)
        y_high = apply_linear(x_high,w,b)
        return y_high - y_low - eps

    @staticmethod
    def constraint_neighbor2(v, x, x_low, x_high):
        w,b = unpack_linear(v)
        y = apply_linear(x, w, b)
        y_low = apply_linear(x_low,w,b)
        y_high = apply_linear(x_high,w,b)
        return y_high - y - eps

    @staticmethod
    def create_constraint_neighbor(x_low, x_high):
        return lambda v: logistic_neighbor.constraint_neighbor(v, x_low, x_high)

    @staticmethod
    def create_constraint_neighbor2(x, x_low, x_high):
        return lambda v: logistic_neighbor.constraint_neighbor2(v, x, x_low, x_high)


class logistic_bound(logistic_optimize):
    @staticmethod
    def eval_mixed_guidance(data, v):
        w, b = unpack_linear(v)
        x = data.x_bound
        bounds = data.bounds
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
        vals2 = -np.log(sig1-sig2 + eps)
        val2 = vals2.sum()
        I = np.isnan(vals2)
        if I.any():
            assert False
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
    def grad_mixed_guidance(data, v):
        bounds = data.bounds
        x = data.x_bound
        w, b = unpack_linear(v)
        y = apply_linear(x, w, b)
        assert y.size == bounds.shape[0]
        c1 = bounds[:, 0]
        c2 = bounds[:, 1]

        sig1 = sigmoid(c2-y)
        sig2 = sigmoid(c1-y)
        denom = sig1 - sig2 + eps
        val = np.zeros(v.shape)
        assert (denom > -1).all()
        for i in range(x.shape[0]):
            num1 = sig1[i]*(1-sig1[i])
            num2 = sig2[i]*(1-sig2[i])
            val[0:-1] += (num1-num2)*(1/denom[i])*x[i,:]
            val[-1] += (num1-num2)*(1/denom[i])
            '''
            num1, x1 = logistic_bound._grad_num_mixed_guidance(x[i, :], c2[i], w, b)
            num2, x2 = logistic_bound._grad_num_mixed_guidance(x[i, :], c1[i], w, b)
            t1 = num1*x1 - num2*x2
            t2 = num1 - num2
            val[0:-1] += (num1*x1 - num2*x2) / denom[i]
            val[-1] += (num1 - num2) / denom[i]
            '''
        #assert not np.isnan(val).any()
        #Why isn't this necessary?
        #val *= -1
        if np.isnan(val).any():
            print 'grad_linear_bound_logistic: nan!'
            val[np.isnan(val)] = 0
        if np.isinf(val).any():
            print 'grad_linear_bound_logistic: inf!'
            val[np.isinf(val)] = 0
        return val


