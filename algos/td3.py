import copy
import numpy as np
import time
import torch
import torch.nn.functional as F
from torch.optim import Adam
import gym
from typing import Optional
import itertools
from utils.util import count_vars
from utils.logx import EpochLogger
from policy.td3_mlp import ActorCritic
from storage.off_policy import ReplayBuffer


def TD3(env_name,
        policy=ActorCritic,
        policy_kwargs: Optional[dict] = None,
        seed: int = 0,
        steps_per_epoch: int = 4000,
        num_epochs: int = 100,
        buffer_size: int = int(1e6),
        gamma: float = 0.99,
        tau: float = 0.005,
        pi_lr: float = 1e-3,
        q_lr: float = 1e-3,
        batch_size: int = 100,
        start_steps: int = 10000,
        update_after: int = 10000,
        update_every: int = 50,
        expl_noise: float = 0.1,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_delay: int = 2,
        max_ep_len: int = 1000,
        num_test_episodes: int = 10,
        logger_kwargs: Optional[dict] = None,
        save_freq=1):
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Instantiate environment
    env = gym.make(env_name)
    test_env = gym.make(env_name)
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape
    max_action = env.action_space.high[0]

    # Create actor-critic module and target networks
    ac = policy(env.observation_space, env.action_space, **policy_kwargs)
    ac_target = copy.deepcopy(ac)

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_target.parameters():
        p.requires_grad = False

    # List of parameters for both Q-networks (save this for convenience)
    q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=buffer_size)

    # Count variables (protip: try to get a feel for how different size networks behave!)
    var_counts = tuple(count_vars(module) for module in [ac.pi, ac.q1, ac.q2])
    logger.log_msg('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n' % var_counts)

    # Set up function for computing TD3 Q-losses
    def compute_critic_loss(data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        # Get current Q estimates
        current_Q1, current_Q2 = ac.q1(o, a), ac.q2(o, a)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            clipped_noise = (torch.randn_like(a) * policy_noise).clamp(-noise_clip, noise_clip)
            a2 = (ac_target.pi(o2) + clipped_noise).clamp(-max_action, max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = ac_target.q1(o2, a2), ac_target.q2(o2, a2)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = r + gamma * (1 - d) * target_Q

        # MSE loss against TD target
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Useful info for logging
        loss_info = dict(Q1Vals=current_Q1.detach().numpy(),
                         Q2Vals=current_Q2.detach().numpy())

        return critic_loss, loss_info

    # Set up function for computing TD3 pi loss
    def compute_actor_loss(data):
        o = data['obs']
        actor_loss = - torch.mean(ac.q1(o, ac.pi(o)))
        return actor_loss

    # def get_action(o, noise_scale):
    #     a = ac.act(torch.as_tensor(o, dtype=torch.float32))
    #     a += noise_scale * np.random.randn(act_dim)
    #     return np.clip(a, -max_action, max_action)

    def get_action(state):
        state = torch.FloatTensor(state)
        action = ac.act(state)
        action += np.random.normal(0, expl_noise, act_dim)
        return np.clip(action, -max_action, max_action)

    # Set up optimizers for policy and q-function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    q_optimizer = Adam(q_params, lr=q_lr)

    def update(data, timer):
        # Compute critic loss
        loss_q, loss_info = compute_critic_loss(data)

        # Optimize the critic
        q_optimizer.zero_grad()
        loss_q.backward()
        q_optimizer.step()

        # Record things
        # logger.store(LossQ=loss_q.item(), **loss_info)
        # logger.store(LossQ=loss_q.item())

        # Delayed policy updates
        if timer % policy_delay == 0:

            # Freeze Q-networks so you don't waste computational effort
            # computing gradients for them during the policy learning step.
            for p in q_params:
                p.requires_grad = False

            # Compute actor loss
            loss_pi = compute_actor_loss(data)

            # Optimize the actor
            pi_optimizer.zero_grad()
            loss_pi.backward()
            pi_optimizer.step()

            # Unfreeze Q-networks so you can optimize it at next DDPG step.
            for p in q_params:
                p.requires_grad = True

            # Record things
            # logger.store(LossPi=loss_pi.item())

            # Update the frozen target models
            for p, p_target in zip(ac.parameters(), ac_target.parameters()):
                p_target.data.copy_(tau * p.data + (1 - tau) * p_target.data)

    def test_agent():
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            while not (d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                o, r, d, _ = test_env.step(get_action(o))
                ep_ret += r
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    # Prepare for interaction with environment
    total_steps = steps_per_epoch * num_epochs
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):

        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards,
        # use the learned policy (with some noise, via act_noise).
        if t > start_steps:
            a = get_action(o)
        else:
            a = env.action_space.sample()

        # Step the env
        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len == max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d)

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        o = o2

        # End of trajectory handling
        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, ep_ret, ep_len = env.reset(), 0, 0

        # Update handling
        if t >= update_after and t % update_every == 0:
            for j in range(update_every):
                batch = replay_buffer.sample_batch(batch_size)
                update(data=batch, timer=j)

        # End of epoch handling
        if (t + 1) % steps_per_epoch == 0:
            epoch = (t + 1) // steps_per_epoch

            # Save model
            if (epoch % save_freq == 0) or (epoch == num_epochs):
                logger.save_policy(state_dict=ac.state_dict())

            # Test the performance of the deterministic version of the agent.
            # test_agent()

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            # logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            # logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            # logger.log_tabular('Q1Vals', with_min_and_max=True)
            # logger.log_tabular('Q2Vals', with_min_and_max=True)
            # logger.log_tabular('LossPi', average_only=True)
            # logger.log_tabular('LossQ', average_only=True)
            logger.log_tabular('Time', time.time() - start_time)
            logger.dump_tabular()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    # ----------------------- env and logger -----------------------
    parser.add_argument('--exp_name', type=str, default='TD3')
    parser.add_argument('--env', type=str, default='LunarLanderContinuous-v2')
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--output_dir', '-o', default=None)

    # ---------------------- hyper-parameters ----------------------
    parser.add_argument('--pi_lr', type=float, default=3e-4)
    parser.add_argument('--vf_lr', type=float, default=3e-4)
    parser.add_argument('--hid', type=int, default=(256, 256))
    parser.add_argument('--gamma', type=float, default=0.99)

    # ------------------------- TD3-Param --------------------------
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--buffer_size', type=int, default=int(1e6))
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_test', type=int, default=10)
    parser.add_argument('--start_steps', type=int, default=int(20e3))
    parser.add_argument('--update_after', type=int, default=10000)
    parser.add_argument('--update_every', type=int, default=50)
    parser.add_argument('--expl_noise', type=float, default=0.1)
    parser.add_argument('--policy_noise', type=float, default=0.2)
    parser.add_argument('--noise_clip', type=float, default=0.5)
    parser.add_argument('--policy_delay', type=int, default=2)

    # ----------------------- training loop ------------------------
    # training parameters
    parser.add_argument('--save_freq', type=int, default=10)
    # epochs
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--max_ep_len', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()

    logger_kwargs = dict(output_dir=args.output_dir,
                         exp_name=args.exp_name,
                         env_name=args.env,
                         seed=args.seed)
    ac_kwargs = dict(hidden_sizes=args.hid)
    td3_kwargs = dict(
        env_name=args.env,
        policy=ActorCritic,
        policy_kwargs=ac_kwargs,
        pi_lr=args.pi_lr,
        q_lr=args.vf_lr,
        gamma=args.gamma,
        tau=args.tau,
        seed=args.seed,
        steps_per_epoch=args.steps,
        num_epochs=args.epochs,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        max_ep_len=args.max_ep_len,
        start_steps=args.start_steps,
        update_every=args.update_every,
        update_after=args.update_after,
        expl_noise=args.expl_noise,
        policy_noise=args.policy_noise,
        policy_delay=args.policy_delay,
        num_test_episodes=args.num_test,
        save_freq=args.save_freq,
        logger_kwargs=logger_kwargs
    )
    TD3(**td3_kwargs)
