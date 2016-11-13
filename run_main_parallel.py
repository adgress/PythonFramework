from subprocess import call, Popen
from timer.timer import tic, toc
import sys
from main import configs_lib
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
from utility.mpi_utility import mpi_rollcall, mpi_split_even_odd
from utility import mpi_group_pool
from mpipool import core as mpipool
import os

comm = MPI.COMM_WORLD
use_mpi = comm.Get_size() > 1
debug_mpi_pool = False
use_multiprocessing_pool = True
parallelize_cv = False

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


mpi_comms = None
def mpi_run_main_args(args):
    #return None
    args = list(args)
    if len(mpi_comms) > 1 or parallelize_cv:
        my_comm = mpi_comms[helper_functions.get_hostname()]
        args.append(my_comm)
    main.run_main_args(args)


def results_exist(configs):
    results_file = configs.results_file
    return os.path.isfile(results_file)

if __name__ == '__main__':
    timer.tic()
    pc = configs_lib.create_project_configs()
    #num_labels_list = num_labels_list[0:10]
    if use_mpi:
        mpi_rollcall()
        '''
        from mpipool import core as mpipool
        pool = mpipool.MPIPool(debug=debug_mpi_pool, loadbalance=True)        
        pool.map(main.run_main_args, num_labels_list)
        #pool.map(launch_subprocess_args, num_labels_list)
        pool.close()
        '''
        mpi_comms = mpi_utility.mpi_comm_by_node(include_root=False)
        if len(mpi_comms) > 1 or parallelize_cv:
            pool = mpi_group_pool.MPIGroupPool(debug=False, loadbalance=True, comms=mpi_comms)
        #assert False
        else:
            pool = mpipool.MPIPool(debug=False, loadbalance=True)
        batch_configs = configs_lib.BatchConfigs(configs_lib.ProjectConfigs())
        comm = MPI.COMM_WORLD
        for c in batch_configs.config_list:
            if results_exist(c):
                if comm.Get_rank() == 0:
                    print 'Skipping: ' + c.results_file
                continue
            if comm.Get_rank() == 0:
                timer.tic()
            num_labels_list = list(itertools.product(c.num_labels, range(c.num_splits)))
            no_viz = False
            pool.map(mpi_run_main_args, [n + (no_viz, c, ) for n in num_labels_list])
            pool.close()

            if comm.Get_rank() == 0:
                print 'TOTAL TIME:'
                timer.toc()
                main.run_main(configs=c)
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

