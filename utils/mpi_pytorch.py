
import torch
from utils.mpi_tools import broadcast, mpi_avg, num_process


def setup_pytorch_for_mpi():
    """
    Avoid slowdowns caused by each separate process's PyTorch using
    more than its fair share of CPU resources.
    """
    if torch.get_num_threads() == 1:
        return
    fair_num_threads = max(int(torch.get_num_threads() / num_process()), 1)
    torch.set_num_threads(fair_num_threads)


def mpi_avg_grads(module):
    """ Average contents of gradient buffers across MPI processes. """
    if num_process() == 1:
        return
    for p in module.parameters():
        # p.grad.copy_(torch.as_tensor(mpi_avg(p.grad)))
        p_grad_numpy = p.grad.numpy()  # numpy view of tensor data
        avg_p_grad = mpi_avg(p.grad)
        p_grad_numpy[:] = avg_p_grad[:]


def sync_params(module):
    """ Sync all parameters of module across all MPI processes. """
    if num_process() == 1:
        return
    for p in module.parameters():
        p_numpy = p.data.numpy()
        broadcast(p_numpy, root=0)


# if __name__ == '__main__':
#     from mpi_tools import mpi_fork
#     from model import ActorCritic
#     import gym
#
#     mpi_fork(4)
#
#     env = gym.make('LunarLander-v2')
#     ac = ActorCritic(env.observation_space, env.action_space, (64, 64))
#
#     setup_pytorch_for_mpi()
#     sync_params(ac)
#
#     for p in ac.actor.parameters():
#         print(f'origin {p.data}')
#         p_numpy = p.data.numpy()  # numpy view of tensor data
#         avg_p = mpi_avg(p.data)
#         p_numpy[:] = avg_p[:]
#
#     for p in ac.actor.parameters():
#         print(f'avg {p.data}')


