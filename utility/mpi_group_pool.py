from mpi4py import MPI
from mpipool.core import _error_function, _function_wrapper, _close_pool_message, MPIPoolException
import traceback
from utility import helper_functions
from utility import mpi_utility

class MPIGroupPool(object):
    """
    A pool that distributes tasks over a set of MPI processes using
    mpi4py. MPI is an API for distributed memory parallelism, used
    by large cluster computers. This class provides a similar interface
    to Python's multiprocessing Pool, but currently only supports the
    :func:`map` method.

    Contributed initially by `Joe Zuntz <https://github.com/joezuntz>`_.

    Parameters
    ----------
    comm : (optional)
        The ``mpi4py`` communicator.

    debug : bool (optional)
        If ``True``, print out a lot of status updates at each step.

    loadbalance : bool (optional)
        if ``True`` and the number of taskes is greater than the
        number of processes, tries to loadbalance by sending out
        one task to each cpu first and then sending out the rest
        as the cpus get done.
    """
    def __init__(self, comm=None, debug=False, loadbalance=False, comms=None):
        self.comm = MPI.COMM_WORLD if comm is None else comm
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size() - 1
        self.debug = debug
        self.function = _error_function
        self.loadbalance = loadbalance
        self.comms = comms
        if not self.is_master() and (comms is None or len(self.comms) != 1):
            raise RuntimeError("Invalid number of group:" + str(len(comms)))
        self.node_name_to_tag = {}

        if self.is_master():
            for i, s in enumerate(comms.keys()):
                self.node_name_to_tag[s] = i

        self.node_name_to_tag = self.comm.bcast(self.node_name_to_tag, root=0)
        self.tag_to_node_name = {v: k for k, v in self.node_name_to_tag.items()}
        self.node_name_to_rank = mpi_utility.mpi_gather_hostnames(comm=self.comm, include_root=False)
        if not self.is_master() and helper_functions.get_hostname() not in self.node_name_to_tag:
            raise RuntimeError("Node " + helper_functions.get_hostname() + " not in node_name_to_tag")
        if self.size == 0:
            raise ValueError("Tried to create an MPI pool, but there "
                             "was only one MPI process available. "
                             "Need at least two.")
        if self.num_groups == 0:
            raise ValueError("Tried to create an MPI pool, but there were no groups available")

    @property
    def num_groups(self):
        return len(self.node_name_to_tag)

    def is_master(self):
        """
        Is the current process the master?

        """
        return self.rank == 0

    def is_group_root(self):
        group_members = self.node_name_to_rank[helper_functions.get_hostname()]
        if 0 in group_members:
            group_members.remove(0)
        return self.rank == min(group_members)

    def get_tag(self):
        if self.is_master():
            raise RuntimeError("Master node doesn't have a tag.")
        return self.node_name_to_tag[helper_functions.get_hostname()]

    def get_comm_for_tag(self, group_id):
        hostname = self.tag_to_node_name[group_id]
        return self.comms[hostname]

    def get_group_members(self, group_id):
        hostname = self.tag_to_node_name[group_id]
        a = self.node_name_to_rank[hostname]
        if 0 in a:
            a.remove(0)
        return a

    def get_tag_for_worker(self, worker_rank):
        s = None
        for hostname, members in self.node_name_to_rank.items():
            if worker_rank in members:
                s = hostname
                break
        assert s is not None
        return self.node_name_to_tag[s]


    def wait(self):
        """
        If this isn't the master process, wait for instructions.

        """
        if self.is_master():
            raise RuntimeError("Master node told to await jobs.")

        status = MPI.Status()

        while True:
            # Event loop.
            # Sit here and await instructions.
            if self.debug:
                print("Worker {0} waiting for task.".format(self.rank))

            # Blocking receive to wait for instructions.
            task = self.comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
            if self.debug:
                print("Group {0}, Worker {1} got task {2} with tag {3}."
                      .format(self.get_tag(), self.rank, task, status.tag))

            # Check if message is special sentinel signaling end.
            # If so, stop.
            if isinstance(task, _close_pool_message):
                if self.debug:
                    print("Worker {0} told to quit.".format(self.rank))
                break

            # Check if message is special type containing new function
            # to be applied
            if isinstance(task, _function_wrapper):
                self.function = task.function
                if self.debug:
                    print("Worker {0} replaced its task function: {1}."
                          .format(self.rank, self.function))
                continue

            # If not a special message, just run the known function on
            # the input and return it asynchronously.
            try:
                result = self.function(task)
            except:
                tb = traceback.format_exc()
                self.comm.isend(MPIPoolException(tb), dest=0, tag=status.tag)
                return
            if self.is_group_root():
                if self.debug:
                    print("Worker {0} sending answer {1} with tag {2}."
                          .format(self.rank, result, status.tag))
                self.comm.isend(result, dest=0, tag=status.tag)


    def map(self, function, tasks, callback=None):
        """
        Like the built-in :func:`map` function, apply a function to all
        of the values in a list and return the list of results.

        Parameters
        ----------
        function : callable
            The function to apply to each element in the list.

        tasks :
            A list of tasks -- each element is passed to the input
            function.

        callback : callable (optional)
            A callback function to call on each result.

        """
        ntask = len(tasks)

        # If not the master just wait for instructions.
        if not self.is_master():
            self.wait()
            return

        if function is not self.function:
            if self.debug:
                print("Master replacing pool function with {0}."
                      .format(function))

            self.function = function
            F = _function_wrapper(function)

            # Tell all the workers what function to use.
            requests = []
            for i in range(self.size):
                r = self.comm.isend(F, dest=i + 1)
                requests.append(r)

            # Wait until all of the workers have responded. See:
            #       https://gist.github.com/4176241
            MPI.Request.waitall(requests)

        if (not self.loadbalance) or (ntask <= self.num_groups):
            # Do not perform load-balancing - the default load-balancing
            # scheme emcee uses.

            # Send all the tasks off and wait for them to be received.
            # Again, see the bug in the above gist.
            requests = []
            for i, task in enumerate(tasks):
                group_id = i % self.num_groups
                if self.debug:
                    print("Sending task {0} to group {1} with tag {2}."
                          .format(task, group_id, i))
                group_members = self.get_group_members(group_id)
                for r in group_members:
                    if self.debug:
                        print("Sent task {0} to worker {1} with tag {2}.".format(task, r, i))
                    r = self.comm.isend(task, dest=r, tag=i)
                requests.append(r)
            MPI.Request.waitall(requests)
            results = []
            for i in range(ntask):
                group_id = i % self.num_groups
                if self.debug:
                    print("Master waiting for group {0} with tag {1}".format(group_id, i))
                group_members = self.get_group_members(group_id)
                #result = self.comm.recv(source=min(group_members), tag=i)
                result = self.comm.recv(source=MPI.ANY_SOURCE, tag=i)
                if isinstance(result, MPIPoolException):
                    print("One of the MPIPool workers failed with the "
                          "exception:")
                    print(result.traceback)
                    raise result

                if callback is not None:
                    callback(result)

                results.append(result)

            return results

        else:
            # Perform load-balancing. The order of the results are likely to
            # be different from the previous case.

            for i, task in enumerate(tasks[0:self.num_groups]):
                group_id = i
                if self.debug:
                    print("Sent task {0} to gruop {1} with tag {2}."
                          .format(task, group_id, i))
                # Send out the tasks asynchronously.
                group_members = self.get_group_members(group_id)
                for r in group_members:
                    self.comm.isend(task, dest=r, tag=i)

            ntasks_dispatched = self.num_groups
            results = [None]*ntask
            for itask in range(ntask):
                status = MPI.Status()
                # Receive input from workers.
                result = self.comm.recv(source=MPI.ANY_SOURCE,
                                        tag=MPI.ANY_TAG, status=status)
                group_root_worker = status.source
                group_id = self.get_tag_for_worker(group_root_worker)
                group_members = self.get_group_members(group_id)
                i = status.tag

                if callback is not None:
                    callback(result)

                results[i] = result
                if self.debug:
                    print("Master received from worker {0} with tag {1}"
                          .format(group_root_worker, i))

                # Now send the next task to this idle worker (if there are any
                # left).
                if ntasks_dispatched < ntask:
                    task = tasks[ntasks_dispatched]
                    i = ntasks_dispatched

                    if self.debug:
                        print("Sent task {0} to group {1} with tag {2}."
                              .format(task, group_id, i))
                    # Send out the tasks asynchronously.
                    for r in group_members:
                        self.comm.isend(task, dest=r, tag=i)
                    ntasks_dispatched += 1

            return results

    def bcast(self, *args, **kwargs):
        """
        Equivalent to mpi4py :func:`bcast` collective operation.
        """
        return self.comm.bcast(*args, **kwargs)

    def close(self):
        """
        Just send a message off to all the pool members which contains
        the special :class:`_close_pool_message` sentinel.

        """
        if self.is_master():
            for i in range(self.size):
                self.comm.isend(_close_pool_message(), dest=i + 1)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
