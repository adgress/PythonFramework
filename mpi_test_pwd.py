import os
from mpi4py import MPI
from mpipool import core as mpipool

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

d = os.getcwd()
print str(rank) + ': ' + d

