from mpi4py import MPI
from mpipool import core as mpipool


def mpipool_test_func(args):
    print str(args[0]) + '-' + str(args[1])


if __name__ == '__main__':
    pool = mpipool.MPIPool(debug=False, loadbalance=True)
    comm = MPI.COMM_WORLD
    for i in range(10):
        args = list(range(20))
        pool.map(mpipool_test_func, [n + (i,) for n in args])
        pool.close()

        if comm.Get_rank() == 0:
            print 'TOTAL TIME:'
