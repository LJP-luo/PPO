import psutil
from policy.on_policy.ppo_mlp import ActorCritic
import argparse
from utils.mpi_tools import mpi_fork
from algos.ppo import PPO


def get_args():
    parser = argparse.ArgumentParser()
    # ----------------------- env and logger -------------------------
    parser.add_argument('--exp_name', type=str, default='PPO')
    parser.add_argument('--env', type=str, default='LunarLander-v2')
    parser.add_argument('--seed', '-s', type=int, default=1000)
    parser.add_argument('--output_dir', '-o', default=None)

    # ---------------------- hyper-parameters ----------------------
    parser.add_argument('--pi_lr', type=float, default=3e-4)
    parser.add_argument('--vf_lr', type=float, default=1e-3)
    parser.add_argument('--hid', type=int, default=(64, 64))
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lam', type=float, default=0.97)
    parser.add_argument('--clip_ratio', type=float, default=0.2)

    # ----------------------- training loop -------------------------
    # decide how many CPUs to use.
    parser.add_argument('--cpu', type=int, default=10)
    # training parameters
    parser.add_argument('--target_kl', type=float, default=0.1)
    parser.add_argument('--train_pi_iter', type=int, default=80)
    parser.add_argument('--train_v_iter', type=int, default=80)
    parser.add_argument('--save_freq', type=int, default=10)
    # epochs
    parser.add_argument('--steps', type=int, default=20000)
    parser.add_argument('--max_ep_len', type=int, default=2000)
    parser.add_argument('--epochs', type=int, default=100)

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    # Determine number of CPU cores to run on
    num_cpu = psutil.cpu_count(logical=False) if args.cpu is None else args.cpu
    mpi_fork(num_cpu)  # run parallel code with mpi

    logger_kwargs = dict(output_dir=args.output_dir,
                         exp_name=args.exp_name,
                         env_name=args.env,
                         seed=args.seed)
    ac_kwargs = dict(hidden_sizes=args.hid)
    ppo_kwargs = dict(
        env_name=args.env,
        policy=ActorCritic,
        policy_kwargs=ac_kwargs,
        pi_lr=args.pi_lr,
        vf_lr=args.vf_lr,
        gamma=args.gamma,
        lam=args.lam,
        clip_ratio=args.clip_ratio,
        seed=args.seed,
        steps_per_epoch=args.steps,
        epochs=args.epochs,
        max_ep_len=args.max_ep_len,
        target_kl=args.target_kl,
        save_freq=args.save_freq,
        logger_kwargs=logger_kwargs
    )
    PPO(**ppo_kwargs)
