import numpy as np
from numpy.linalg import eig
from numpy.linalg import inv
from numpy.linalg import norm
def make_random_psd(n=50, p=10):
    A = np.random.rand(n,p)
    A = A.T.dot(A)
    w,v = eig(A)
    assert (w >= 0).all()
    return A

def f(X):
    M = inv(X + .000001*np.eye(X.shape[0]))
    #return np.trace(M.dot(X))
    w, v = eig(M.dot(X))
    w_M, _ = eig(M)
    w_X, _ = eig(X)
    w.sort()
    w_M.sort()
    w_X.sort()

    print w[-5] - w_X[-5] * w_M[4]
    return w.max()

def rel_err(x, y):
    return norm(x-y)/norm(x)


def test_bound_fro():
    A = make_random_psd()
    B = make_random_psd()
    '''
    f1 = norm(A+B,'fro')**2
    f2 = norm(A,'fro')**2 + norm(B,'fro')**2
    '''
    f1 = norm(A + B) ** 2
    f2 = norm(A) ** 2 + norm(B) ** 2
    assert f1 >= f2

def test_bound():
    A = make_random_psd()
    B = make_random_psd()
    f1 = f(.5*A + .5*B)
    f2 = .5*(f(A) + f(B))
    r = rel_err(f1,f2)
    #assert f1 >= f2


if __name__ == '__main__':
    for i in range(10000):
        test_bound_fro()