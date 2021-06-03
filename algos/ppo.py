import numpy as np
import torch
from torch.optim import Adam
import gym
import time
from typing import Optional
import psutil

from policy.ppo_mlp import ActorCritic
from utils.util import count_vars
from utils.logx import EpochLogger
from utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from utils.mpi_tools import mpi_avg, process_id, num_process, mpi_fork
from storage.on_policy import RolloutBuffer


def PPO(env_name,
        policy=ActorCritic,
        policy_kwargs: Optional[dict] = None,
        seed: int = 0,
        steps_per_epoch: int = 4000,
        epochs: int = 50,
        max_ep_len: int = 1000,
        gamma: float = 0.99,
        lam: float = 0.97,
        clip_ratio: float = 0.2,
        pi_lr: float = 3e-4,
        vf_lr: float = 1e-3,
        train_pi_iters: int = 80,
        train_v_iters: int = 80,
        target_kl: float = 0.1,
        logger_kwargs: Optional[dict] = None,
        save_freq: int = 10) -> None:
    # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    setup_pytorch_for_mpi()

    # Set up logger and save configuration
    logger = EpochLogger(**logger_kwargs)
    # Return a dictionary representing the current local symbol table, and save in config.json file.
    logger.save_config(locals())

    # Random seed
    seed += 10000 * process_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Instantiate environment
    env = gym.make(env_name)
    env.seed(seed)
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    # Create actor-critic module
    ac = policy(env.observation_space, env.action_space, **policy_kwargs)

    # Sync params across processes
    sync_params(ac)

    # Count variables
    var_counts = tuple(count_vars(module) for module in [ac.actor, ac.critic])
    logger.log_msg('\nNumber of parameters: \t pi: %d, \t v: %d\n' % var_counts)

    # Set up experience buffer
    local_steps_per_epoch = int(steps_per_epoch / num_process())
    buf = RolloutBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)

    # Set up function for computing PPO policy loss
    def compute_loss_pi(data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

        # Policy loss
        pi, logp = ac.actor(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1 + clip_ratio) | ratio.lt(1 - clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    # Set up function for computing value loss
    def compute_loss_v(data):
        obs, ret = data['obs'], data['ret']
        return ((ac.critic(obs) - ret) ** 2).mean()

    # Set up optimizers for policy and value function
    pi_optimizer = Adam(ac.actor.parameters(), lr=pi_lr)
    vf_optimizer = Adam(ac.critic.parameters(), lr=vf_lr)

    def update():
        data = buf.get()

        pi_l_old, pi_info_old = compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = compute_loss_v(data).item()

        # Train policy with multiple steps of gradient descent
        for i in range(train_pi_iters):
            pi_optimizer.zero_grad()
            loss_pi, pi_info = compute_loss_pi(data)
            kl = mpi_avg(pi_info['kl'])
            if kl > 1.5 * target_kl:
                logger.log_msg(f'Early stopping at step {i} due to reaching max kl.')
                break
            loss_pi.backward()
            mpi_avg_grads(ac.actor)  # average grads across MPI processes
            pi_optimizer.step()

        logger.store(StopIter=i)

        # Value function learning
        for i in range(train_v_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            mpi_avg_grads(ac.critic)  # average grads across MPI processes
            vf_optimizer.step()

        # Log changes from update
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        logger.store(LossPi=pi_l_old, LossV=v_l_old,
                     KL=kl, Entropy=ent, ClipFrac=cf,
                     DeltaLossPi=(loss_pi.item() - pi_l_old),
                     DeltaLossV=(loss_v.item() - v_l_old))

    # Prepare for interaction with environment
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        for t in range(local_steps_per_epoch):
            a, v, logp = ac.step(o)

            next_o, r, d, _ = env.step(a)
            ep_ret += r
            ep_len += 1

            # save and log
            buf.store(o, a, r, v, logp)
            logger.store(VVals=v)

            # Update obs (critical!)
            o = next_o

            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = t == local_steps_per_epoch - 1

            if terminal or epoch_ended:
                if epoch_ended and not terminal:
                    print(f'Process_{process_id()} Warning: trajectory cut off '
                          f'by epoch at {ep_len} steps.', flush=True)
                # if trajectory didn't reach terminal state, bootstrap value target
                if timeout or epoch_ended:
                    _, v, _ = ac.step(o)
                else:
                    v = 0
                buf.finish_path(v)
                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    logger.store(EpRet=ep_ret, EpLen=ep_len)
                o, ep_ret, ep_len = env.reset(), 0, 0

        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs - 1):
            logger.save_policy(state_dict=ac.state_dict())
            logger.log_msg(msg='Model has been saved!', color='green')

        # Perform PPO update!
        update()

        # Log info about epoch
        logger.log_tabular('Epoch', epoch + 1)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch + 1) * steps_per_epoch)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('ClipFrac', average_only=True)
        logger.log_tabular('StopIter', average_only=True)
        logger.log_tabular('Time', time.time() - start_time)
        # print and save epoch info
        logger.dump_tabular()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    # ----------------------- env and logger -------------------------
    parser.add_argument('--exp_name', type=str, default='PPO')
    parser.add_argument('--env', type=str, default='Ant-v2')
    parser.add_argument('--seed', '-s', type=int, default=0)
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
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--max_ep_len', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()

    # Determine number of CPU cores to run on
    num_cpu = psutil.cpu_count(logical=False) if args.cpu is None else args.cpu
    mpi_fork(num_cpu)  # run parallel code with mpi

    logger_kwargs = dict(output_dir=args.output_dir, exp_name=args.exp_name, env_name=args.env)
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
