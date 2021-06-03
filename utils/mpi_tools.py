from mpi4py import MPI
import os
import subprocess
import sys
import numpy as np


def mpi_fork(n, bind_to_core=False):
    if n <= 1:
        return
    if os.getenv("IN_MPI") is None:
        env = os.environ.copy()
        env.update(
            MKL_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
            IN_MPI="1"
        )
        args = ["mpiexec", "-n", str(n)]
        if bind_to_core:
            args += ["-bind-to", "core"]
        # find an executable python and
        args += [sys.executable] + sys.argv
        print(args)
        subprocess.check_call(args, env=env)
        sys.exit()


def msg(m, string=''):
    print(f'Message from {MPI.COMM_WORLD.Get_rank()}: {string + str(m)}')


def process_id():
    return MPI.COMM_WORLD.Get_rank()


def num_process():
    return MPI.COMM_WORLD.Get_size()


def allreduce(*args, **kwargs):
    return MPI.COMM_WORLD.Allreduce(*args, **kwargs)


def broadcast(x, root=0):
    MPI.COMM_WORLD.Bcast(x, root=root)


def mpi_op(x, op):
    x, scalar = ([x], True) if np.isscalar(x) else (x, False)
    x = np.asarray(x, dtype=np.float32)
    buff = np.zeros_like(x, dtype=np.float32)
    # communication
    # root node receives results with a collective "reduce"
    allreduce(sendbuf=x, recvbuf=buff, op=op)
    return buff[0] if scalar else buff


def mpi_sum(x):
    return mpi_op(x, op=MPI.SUM)


def mpi_avg(x):
    return mpi_sum(x) / num_process()


def mpi_statistics_scalar(x, with_min_and_max=False):
    """
    Get mean/std and optional min/max of scalar x across MPI processes.
    Args:
        x: An array containing samples of the scalar to produce statistics for.
        with_min_and_max (bool): If true, return min and max of x in addition to mean and std.
    """
    x = np.array(x, dtype=np.float32)
    global_sum, global_n = map(mpi_sum, [np.sum(x), len(x)])
    # global_sum, global_n = mpi_sum([np.sum(x), len(x)])
    mean = global_sum / global_n

    global_sum_sq = mpi_sum(np.sum(x - mean) ** 2)
    # compute global std
    std = np.sqrt(global_sum_sq / global_n)

    if with_min_and_max:
        global_min = mpi_op(np.min(x) if len(x) > 0 else np.inf, op=MPI.MIN)
        global_max = mpi_op(np.max(x) if len(x) > 0 else np.inf, op=MPI.MAX)
        return mean, std, global_min, global_max
    return mean, std


# if __name__ == '__main__':
#     mpi_fork(4)
#     rank = process_id()
#     size = num_process()
#     x = np.array([rank] * 10, dtype=np.float32)
#     data_2 = np.array([rank] * 4, dtype=np.float32)
#     # s_1, s_2 = mpi_sum([data_1, data_2])
#     # s_1, s_2 = map(mpi_sum, [data_1, data_2])
#     mean, std = mpi_statistics_scalar(x)
#     # global_sum, global_n = mpi_sum([np.sum(x), len(x)])
#     # mean = global_sum / global_n
#     #
#     # global_sum_sq = mpi_sum(np.sum(x - mean) ** 2)
#     # # compute global std
#     # std = np.sqrt(global_sum_sq / global_n)
#     print(mean, std)
#     # print(mpi_avg(np.sum(x)))
