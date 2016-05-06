from subprocess import call, Popen
from timer.timer import tic, toc
import sys
import active.active_project_configs as configs_lib
import itertools
from utility import multiprocessing_utility
from utility import helper_functions
import main
import multiprocessing
from mpi4py import MPI
import numpy as np
from timer import timer
import inspect
from utility import mpi_utility
from utility import mpi_group_pool

comm = MPI.COMM_WORLD
use_mpi = comm.Get_size() > 1
debug_mpi_pool = False
use_multiprocessing_pool = True

if helper_functions.is_laptop():
    pool_size = 2
else:
    pool_size = 24

def launch_subprocess_args(args):
    #print args
    #return
    launch_subprocess(*args)

def launch_subprocess(num_labels, split_idx):
     #sys.argv = ['C:/Users/Aubrey/Desktop/PythonFramework/main.py']
     #multiprocessing.set_executable('C:/Python27/python.exe')
     #multiprocessing.forking.set_executable('C:/Python27/python.exe')
     sys.original_argv = sys.argv
     p = call(['python',
                'main.py',
                '-num_labels', str(num_labels),
                '-split_idx', str(split_idx),
                '-no_viz'
                ])

def mpi_rollcall():
    comm = MPI.COMM_WORLD
    s = comm.Get_size()
    rank = comm.Get_rank()
    for i in range(s):
        if rank == i:
            hostname = helper_functions.get_hostname()
            print '(' + hostname + '): ' + str(rank) + ' of ' + str(s)
        comm.Barrier()

def mpi_split_even_odd():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    color = rank % 2
    local_comm = MPI.COMM_WORLD.Split(color, rank)
    print 'World Rank = ' + str(rank) + ', Local Rank = ' + str(local_comm.Get_rank())
    exit()

mpi_comms = None
def mpi_run_main_args(args):
    #return None
    my_comm = mpi_comms[helper_functions.get_hostname()]
    args = list(args)
    args.append(my_comm)
    main.run_main_args(args)


if __name__ == '__main__':
    timer.tic()
    pc = configs_lib.create_project_configs()
    num_labels_list = list(itertools.product(pc.num_labels, range(pc.num_splits)))
    #num_labels_list = num_labels_list[0:10]
    if use_mpi:
        mpi_rollcall()
        num_labels_list = [i + (True,) for i in num_labels_list]
        '''
        from mpipool import core as mpipool
        pool = mpipool.MPIPool(debug=debug_mpi_pool, loadbalance=True)        
        pool.map(main.run_main_args, num_labels_list)
        #pool.map(launch_subprocess_args, num_labels_list)
        pool.close()
        '''
        mpi_comms = mpi_utility.mpi_comm_by_node(include_root=False)
        pool = mpi_group_pool.MPIGroupPool(debug=False, loadbalance=True, comms=mpi_comms)
        #assert False
        pool.map(mpi_run_main_args, num_labels_list)
        pool.close()
    else:
        if use_multiprocessing_pool:
            pool = multiprocessing_utility.LoggingPool(processes=pool_size)
            pool.map(launch_subprocess_args, num_labels_list)
        else:
            for i in num_labels_list:
                launch_subprocess_args(i)
    comm = MPI.COMM_WORLD
    if comm.Get_rank() == 0:
        print 'TOTAL TIME:'
        timer.toc()
        main.run_main()
