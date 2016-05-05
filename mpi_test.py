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
from utility import helper_functions

mpi_groups = None
mpi_comms = None
def test_mpi_func(i):

    local_comm = mpi_comms[helper_functions.get_hostname()]

    local_comm.Barrier()
    if local_comm.Get_rank() == 0:
        mpi_utility.mpi_print('My Group: ' + str(i),local_comm)
    local_comm.Barrier()

    #mpi_utility.mpi_print(str(i))
    return 'Hello World'

if __name__ == '__main__':
    mpi_groups = mpi_utility.mpi_group_by_node(include_root=False)
    mpi_comms = mpi_utility.mpi_comm_by_node(include_root=False)
    pool = mpi_group_pool.MPIGroupPool(debug=False, loadbalance=True, comms=mpi_groups)
    data = list(range(10))
    pool.map(test_mpi_func, data)
    pool.close()

