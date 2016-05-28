import numpy as np
from numpy.linalg import norm
import scipy.optimize as optimize
from scipy.special import expit as sigmoid

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

def eval_linear_bound_logistic(x, bounds, v):
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

def _grad_num(x, c, w, b=None):
    if b is None:
        w,b = unpack_linear(w)
    a = c - apply_linear(x, w, b)
    t = 1 + np.exp(-a)
    t = t ** -2
    t2 = np.exp(-a)
    return t*t2, x

def grad_linear_bound_logistic(x, bounds, v):
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
        num1, x1 = _grad_num(x[i,:], c2[i], w, b)
        num2, x2 = _grad_num(x[i,:], c1[i], w, b)
        val[0:-1] += (num1*x1 - num2*x2) / denom[i]
        val[-1] += (num1 - num2) / denom[i]
    assert not np.isnan(val).any()
    #Why isn't this necessary?
    #val *= -1
    if np.isnan(val).any():
        print 'nan!'
        val[np.isnan(val)] = 0

    return val

def eval_linear_loss_bound_logistic(x, y, x_bound, bounds, reg_w, reg_neighbor, v):
    w, b = unpack_linear(v)
    y_pred = apply_linear(x, w, b)
    loss = eval_linear_loss_l2(x, y, v)
    loss_neighbor = eval_linear_bound_logistic(x_bound, bounds, v)
    loss_reg = eval_reg_l2(w)

    val = loss + reg_w*loss_reg
    if reg_neighbor != 0:
        val += reg_neighbor*loss_neighbor
    return val

def grad_linear_loss_bound_logistic(x, y, x_bound, bounds, reg_w, reg_neighbor, v):
    w, b = unpack_linear(v)
    grad_loss = grad_linear_loss_l2(x, y, v)
    grad_neighbor = grad_linear_bound_logistic(x_bound, bounds, v)
    grad_reg = grad_reg_l2(w)
    grad_reg *= reg_w
    grad_reg = np.append(grad_reg, 0)

    val = grad_loss + grad_reg
    if reg_neighbor != 0:
        val += reg_neighbor*grad_neighbor
    return val

def create_eval_linear_loss_bound_logistic(x, y, x_bound, bounds, reg_w, reg_neighbor):
    return lambda v: eval_linear_loss_bound_logistic(x, y, x_bound, bounds, reg_w, reg_neighbor, v)

def create_grad_linear_loss_bound_logistic(x, y, x_bound, bounds, reg_w, reg_neighbor):
    return lambda v: grad_linear_loss_bound_logistic(x, y, x_bound, bounds, reg_w, reg_neighbor, v)

def constraint_bound(v, x, upper_bound):
    w,b = unpack_linear(v)
    y = apply_linear(x, w, b)
    return upper_bound - y

def create_constraint_bound(x, upper_bound):
    return lambda v: constraint_bound(v, x, upper_bound)