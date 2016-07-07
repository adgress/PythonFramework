import numpy as np
import cvxpy as cvx
from timer.timer import tic, toc

def run_test():
    num_iterations = 100
    size = (1000,1000)
    X = np.random.uniform(-1,1,size)
    C = 1e-3
    for i in range(num_iterations):
        print str(i) + ' of ' + str(num_iterations)
        inv_test(X, C)

def inv_test(X, C):
    XX = X.T.dot(X) + C*np.eye(X.shape[1])
    np.linalg.inv(XX)

if __name__ == '__main__':
    tic()
    run_test()
    toc()