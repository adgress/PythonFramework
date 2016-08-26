import numpy as np
import cvxpy as cvx
from timer.timer import tic, toc
from mpi4py import MPI
from utility import mpi_utility
from utility import mpi_group_pool
from mpipool import core as mpipool

def run_test():
    num_iterations = 100
    pool = None
    comm = MPI.COMM_WORLD
    if comm.Get_size() > 1:
        #pool = mpi_group_pool.MPIGroupPool(debug=False, loadbalance=True, comms=mpi_comms)
        pool = mpipool.MPIPool(debug=False, loadbalance=True)
    f = normal_test
    #f = cvx_test
    #f = mult_test
    #f = inv_test
    if pool is None:
        for i in range(num_iterations):
            print str(i) + ' of ' + str(num_iterations)
            f()
    else:
        args = [(i,) for i in range(num_iterations)]
        pool.map(f, args)
        pool.close()
        pass

def mult_test(*args):
    print 'mult'
    size = (2000, 2000)
    X = np.random.uniform(-1, 1, size)
    tic()
    XX = X.T.dot(X)
    toc()

def inv_test(*args):
    size = (1000, 1000)
    X = np.random.uniform(-1, 1, size)
    C = 1e-3
    XX = X.T.dot(X) + C*np.eye(X.shape[1])
    np.linalg.inv(XX)

def normal_test(*args):
    n = 5000
    p = 2000
    X = np.random.uniform(-1, 1, (n, p))
    C = 1e-3
    y = np.random.uniform(-1, 1, n)
    tic()
    A = X.T.dot(X) + C*np.eye(p)
    k = X.T.dot(y)
    w = np.linalg.solve(A, k)
    toc()

def cvx_test(*args):
    n = 5000
    p = 100
    X = np.random.uniform(-1,1,(n,p))
    C = 1e-3
    y = np.random.uniform(-1,1, n)
    w = cvx.Variable(p)
    loss = cvx.sum_entries(cvx.square(X*w - y))
    reg = cvx.norm2(w)**2
    obj = cvx.Minimize(loss + C*reg)
    prob = cvx.Problem(obj, [])
    tic()
    prob.solve(solver=cvx.SCS, verbose=False)
    toc()



if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    is_master = comm.Get_rank() == 0
    if is_master:
        tic()
    run_test()
    if is_master:
        toc()