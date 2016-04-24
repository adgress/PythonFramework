import sys
print sys.executable
from mpi4py import MPI
import mpi4py
import numpy as np
from utility import multiprocessing_utility
import multiprocessing
from mpipool import core as mpipool

lock = None
work_left = None
TAG_WORK = 1
TAG_DONE = 2

def manage_work(work_left):

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    i = np.zeros(1)
    not_working = set(range(1,size))
    while True:
        if len(not_working) == 0:
            d = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)
            not_working.add(d['id'])
        if len(work_left) == 0:
            break
        data = {
            'x': work_left.pop(),
            'done': False,
        }
        id = not_working.pop()
        comm.send(data, dest=id, tag=TAG_WORK)
    data = {
        'done': True
    }
    for i in range(1,size):
        comm.send(data, dest=i, tag=TAG_WORK)


def do_work():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    while True:
        data = comm.recv(source=0, tag=TAG_WORK)
        if data['done']:
            break
        print str(rank) + ': ' + str(data['x']) + ',' + str(data['x']**2)
        data['id'] = rank
        comm.send(data, dest=0, tag=TAG_WORK)


def mpi_test():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    #print 'Rank: ' + str(rank)
    #print 'Size: ' + str(size)


    if rank == 0:
        #print 'rank: ' + str(rank)

        lock = multiprocessing.Lock()
        pool = multiprocessing_utility.LoggingPool(processes=size-1)
        all_work = list(range(50))
        work_left = all_work
        args = list(range(1, size))
        #pool.apply(manage_work, args)
        manage_work(work_left)
        print 'rank 0 done'
    else:
        print 'rank: ' + str(rank)
        do_work()

def mpi_pool_work(i):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    print str(rank) + ': ' + str(i) + ',' + str(i**2)

def mpi_pool_test():
    all_work = list(range(1))
    pool = mpipool.MPIPool(debug=True, loadbalance=False)
    pool.map(mpi_pool_work, all_work)
    pool.close()

if __name__ == '__main__':
    mpi_pool_test()

