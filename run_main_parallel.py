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


def mpi_split_by_node():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    hostname = helper_functions.get_hostname()
    data = {
        'rank': rank,
        'hostname': hostname
    }
    data = comm.gather(data,root=0)
    data = comm.bcast(data,root=0)
    if rank == 0:
        all_hostnames = {d['hostname'] for d in data}
        root_nodes = {}
        colors = {}
        color_idx = 0
        for host in all_hostnames:
            all_nodes = np.asarray([d['rank'] for d in data])
            root_nodes[host] = all_nodes.min()
            colors[host] = color_idx
            color_idx += 1
        #print root_nodes
        #print colors
    else:
        root_nodes = None
        colors = None
    root_nodes = comm.bcast(root_nodes,root=0)
    colors = comm.bcast(colors,root=0)
    split_comm = comm.Split(colors[hostname], rank)
    return split_comm


def mpi_test_func(i, comm):
    print str(comm.Get_rank()) + ': ' + str(i)

def mpi_test_func_args(args):
    mpi_test_func_args(*args)

if __name__ == '__main__':
    timer.tic()
    pc = configs_lib.create_project_configs()
    num_labels_list = list(itertools.product(pc.num_labels, range(pc.num_splits)))
    if use_mpi or True:
        mpi_rollcall()
        '''
        mpi_split_by_node()

        from utility import mpi_group_pool
        pool = mpi_group_pool.MPIGroupPool(debug=False, loadbalance=True)
        input = [[i, MPI.COMM_WORLD] for i in range(10)]
        print num_labels_list
        pool.map(mpi_test_func_args, input)
        pool.close
        exit()
        '''
        from mpipool import core as mpipool
        pool = mpipool.MPIPool(debug=debug_mpi_pool, loadbalance=True)
        num_labels_list = [i + (True,) for i in num_labels_list]
        pool.map(main.run_main_args, num_labels_list)
        #pool.map(launch_subprocess_args, num_labels_list)
        pool.close()
    else:
        if use_multiprocessing_pool:
            pool = multiprocessing_utility.LoggingPool(processes=pool_size)
            pool.map(launch_subprocess_args, num_labels_list)
        else:
            for i in num_labels_list:
                launch_subprocess_args(i)
    print 'TOTAL TIME:'
    timer.toc()
    comm = MPI.COMM_WORLD
    if comm.Get_rank() == 0:
        main.run_main()
