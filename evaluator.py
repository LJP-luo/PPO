import json
import os
import torch
import gym
import time
from policy import td3_mlp, ppo_mlp
from utils.logx import EpochLogger


def load_data_from_json(logdir):
    config_path = ''
    data = None
    for root, _, files in os.walk(logdir):
        if 'config.json' in files:
            config_path = os.path.join(root, 'config.json')
            with open(config_path) as f:
                data = json.load(f)

    print(f'Loading config info from:\t {config_path}...\n')

    env_name = data['env_name']
    model_name = data['policy']
    hidden_sizes = data['policy_kwargs']['hidden_sizes']
    exp_name = data['exp_name']

    output_dir = None
    for key, val in data['logger'].items():
        if val['output_dir'] is not None:
            output_dir = val['output_dir']

    assert output_dir, 'Warning, please check out if the output_dir in .json' \
                       ' matches current experiment file.'

    return env_name, model_name, hidden_sizes, exp_name, output_dir


def load_policy(logdir):
    """ Load a pytorch policy saved with Spinning Up Logger."""

    env_name, model_name, hidden_sizes, exp_name, output_dir = \
        load_data_from_json(logdir)
    model_file = os.path.join(output_dir, 'model', f'{exp_name}-actor_critic.pth')
    print(f'Loading state_dict from:\t {model_file}...\n')

    env = gym.make(env_name)
    model = None
    if model_name == 'ActorCritic' and exp_name == 'TD3':
        model = td3_mlp.ActorCritic(env.observation_space, env.action_space, hidden_sizes)
    elif model_name == 'ActorCritic' and exp_name == 'PPO':
        model = ppo_mlp.ActorCritic(env.observation_space, env.action_space, hidden_sizes)
    model.load_state_dict(torch.load(model_file))

    # make function for producing an action given a single state
    def get_action(x):
        with torch.no_grad():
            x = torch.as_tensor(x, dtype=torch.float32)
            action = model.act(x)
        return action

    logger = EpochLogger(output_dir=output_dir, output_fname='evaluation.txt')

    return env, get_action, logger


def run_policy(logdir, max_ep_len=1000, num_episodes=4, render=True):
    env, get_action, logger = load_policy(logdir)

    for ep in range(num_episodes):
        o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
        if render:
            env.render()
            time.sleep(1)
        for t in range(max_ep_len):
            if render:
                env.render()
            a = get_action(o)
            o, r, d, _ = env.step(a)
            ep_ret += r
            ep_len += 1

            if d or (ep_len == max_ep_len - 1):
                logger.log_tabular('Epoch', ep + 1)
                logger.log_tabular('EpRet', round(ep_ret, 2))
                logger.log_tabular('EpLen', ep_len)
                logger.dump_tabular()
                # print(f'Episode {ep + 1} \t EpRet {ep_ret:.3f} \t EpLen {ep_len}')
                break
    env.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, default='experiments/PPO_Ant-v2/seed_0',
                        help='E.g. experiments/PPO_HalfCCheetah-v2/seed_x')

    parser.add_argument('--episodes', '-n', type=int, default=4)
    parser.add_argument('--max_ep_len', type=int, default=1000)
    parser.add_argument('--render', '-r', default=True)
    args = parser.parse_args()

    run_policy(args.logdir, args.max_ep_len, args.episodes, args.render)
