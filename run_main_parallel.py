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

comm = MPI.COMM_WORLD
use_mpi = comm.Get_size() > 1
debug_mpi_pool = False
use_multiprocessing_pool = True

if helper_functions.is_laptop():
    pool_size = 2
else:
    pool_size = 13

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

if __name__ == '__main__':

    pc = configs_lib.create_project_configs()
    num_labels_list = list(itertools.product(pc.num_labels, range(pc.num_splits)))

    if use_mpi:
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
    if MPI.Get_rank() == 0:
        main.run_main()