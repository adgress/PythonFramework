import sys
print sys.executable
from mpi4py import MPI
import mpi4py
import numpy as np
from utility import multiprocessing_utility
import multiprocessing
from mpipool import core as mpipool
from utility import mpi_utility
from utility import mpi_group_pool


def test_mpi_func(i):
    mpi_utility.mpi_print(str(i))
    return 'Hello World'

if __name__ == '__main__':
    mpi_groups = mpi_utility.mpi_group_by_node(include_root=False)
    pool = mpi_group_pool.MPIGroupPool(debug=True, loadbalance=False, groups=mpi_groups)
    data = list(range(10))
    pool.map(test_mpi_func, data)
    pool.close()

