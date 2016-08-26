from mpi4py import MPI
from utility import helper_functions

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

def get_comm():
    import main
    comm = main.configs_lib.comm
    if comm is None:
        comm = MPI.COMM_WORLD
    return comm

def is_group_master():
    comm = get_comm()
    return comm == MPI.COMM_WORLD or comm.Get_rank() == 0

def is_master():
    comm = get_comm()
    rank = comm.Get_rank()
    return rank == 0 or (comm == MPI.COMM_WORLD and rank == 1)

def mpi_print(s, comm=MPI.COMM_WORLD):
    print '(' + str(comm.Get_rank()) + '): ' + str(s)

def mpi_gather_hostnames(comm=MPI.COMM_WORLD, include_root=False):
    hostname = helper_functions.get_hostname()
    all_hostnames = comm.gather(hostname, root=0)
    all_hostnames = comm.bcast(all_hostnames, root=0)
    host_to_rank = {}
    for i, s in enumerate(all_hostnames):
        if not include_root and i == 0:
            continue
        if s not in host_to_rank:
            host_to_rank[s] = set()
        host_to_rank[s].add(i)
    host_to_rank = comm.bcast(host_to_rank, root=0)
    return host_to_rank

def mpi_group_by_node(include_root=False):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    hostnames_to_rank = mpi_gather_hostnames(include_root=include_root)
    all_groups = {}
    for hostname, workers in hostnames_to_rank.items():
        world_group = comm.Get_group()
        local_group = world_group.Incl(list(workers))
        hostnames_to_rank[hostname] = local_group
        if rank == 0:
            mpi_print('Group for ' + hostname + ' created with members: ' + str(list(workers)))
        all_groups[hostname] = local_group
        comm.Barrier()
    return all_groups

def mpi_comm_by_node(include_root=False):
    comm = MPI.COMM_WORLD
    all_groups = mpi_group_by_node(include_root)
    all_comms = {}
    for hostname, group in all_groups.items():
        all_comms[hostname] = comm.Create(group)
    return all_comms
