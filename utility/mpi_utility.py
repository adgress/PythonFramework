from mpi4py import MPI
from utility import helper_functions

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
        if rank == 01:
            mpi_print('Group for ' + hostname + ' created with members: ' + str(list(workers)))
        all_groups[hostname] = local_group
        comm.Barrier()
    return all_groups


def mpi_split_by_node():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    hostname = helper_functions.get_hostname()
    data = {
        'rank': rank,
        'hostname': hostname
    }
    data = comm.gather(data,root=0)
    data = comm.bcast(data,root=0)
    if rank == 0:
        all_hostnames = list({d['hostname'] for d in data})
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
        all_hostnames = None
    root_nodes = comm.bcast(root_nodes,root=0)
    colors = comm.bcast(colors,root=0)
    all_hostnames = comm.bcast(all_hostnames,root=0)
    all_comms = {}
    for s in all_hostnames:
        curr_nodes = None
        if rank == 0 or s == hostname:
            curr_nodes = rank
        if rank == 1:
            curr_nodes = None
        curr_nodes = comm.alltoall([curr_nodes]*size)
        curr_nodes = [i for i in curr_nodes if i is not None]
        world_group = comm.Get_group()
        local_group = world_group.Incl(curr_nodes)

        if rank in curr_nodes:
            local_comm = comm.Create(local_group)
            #print str(rank) + ': ' + str(local_comm.Get_rank()) + ' of ' + str(local_comm.Get_size())
            mpi_print(rank, local_comm)
            all_comms[s] = local_comm
        else:
            #print inspect.getmembers(local_group)
            local_comm = None
        #if local_comm != MPI.MPI_COMM_NULL:
        comm.Barrier()
    return all_comms