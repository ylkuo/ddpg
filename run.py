# Requires GlfwContext so GLEW won't complain
from mujoco_py import GlfwContext
GlfwContext(offscreen=True) # Create a window to init GLFW.

from common.cmd_util import make_vec_env
from ddpg import DDPG
from her_sampler import make_sample_her_transitions
from mpi4py import MPI
from rollout import RolloutWorker
from torch.utils.tensorboard import SummaryWriter

import argparse
import GPUtil
import gym
import numpy as np
import os
import random
import shutil
import torch


PARAMS = {
    'lr': 0.001,              # learning rate of the agent
    # ddpg
    'buffer_size': int(1e6),  # size of replay buffer for experience replay
    'polyak': 0.95,           # polyak averaging coefficient
    'clip_obs': 200.,         # clip observations before normalization to be in [-clip_obs, clip_obs], probably for outlier removal
    'batch_size': 256,        # batch size per mpi thread
    'T': 50,                  # maximum steps an agent can take in a rollout
    'gamma': 0.98,            # discounting factor for future reward estimation
    'clip_return': 50.,       # clip returns to be in [-clip_return, clip_return]
    # training
    'n_cycles': 10,           # number of rollouts per epoch
    'n_batches': 40,          # number of training batches per rollout
    'n_test_rollouts': 10,    # number of rollouts in test
    # exploration
    'random_eps': 0.3,        # percentage of time a random action is taken
    'noise_eps': 0.2,         # std of gaussian noise added to not-completely-random actions
    # normalization
    'norm_eps': 0.01,         # epsilon used for observation normalization
    'norm_clip': 5,           # normalized observations are clipped to this values
}


def set_seed(seed):
    """Set the random seed or get one if it is not given"""
    if not seed:
        files = os.listdir('runs/')
        if not files:
            seed = 0
        else:
            seed = max([int(f.split('seed=')[1][0]) for f in files]) + 1
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    return seed


def choose_gpu(threshold=0.80):
    """Automatically choose the most available GPU"""
    gpus = GPUtil.getGPUs()
    gpus = [gpu for gpu in gpus if gpu.load < threshold and gpu.memoryUtil < threshold]
    gpu = random.choice(gpus).id
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)


def parse_args():
    """Parse the input arguments"""
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--env', help='OpenAI gym Fetch env id', type=str,
                            choices=['FetchReach-v1', 'FetchPush-v1', 'FetchPickAndPlace-v1'])
    arg_parser.add_argument('--num_timesteps', help='Total number of time steps for training',
                            type=float, default=1e6),
    arg_parser.add_argument('--num_workers', help='Number of workers to run the rollouts',
                            default=None, type=int)
    arg_parser.add_argument('--seed', help='Random seed', type=int, default=None)
    args = arg_parser.parse_args()
    PARAMS['num_workers'] = args.num_workers
    PARAMS['num_timesteps'] = args.num_timesteps
    return args


def get_dims(env):
    """Get dimensions of observations, action, and goal from the input env"""
    env_name = env.spec.id
    tmp_env = gym.make(env_name)
    tmp_env.reset()
    obs, _, _, _ = tmp_env.step(env.action_space.sample())
    dims = {
        'o': obs['observation'].shape[0],
        'u': env.action_space.shape[0],
        'g': obs['desired_goal'].shape[0],
        'info_is_success': 1,
    }
    PARAMS['distance_threshold'] = tmp_env.env.distance_threshold
    PARAMS['dims'] = dims


def train(policy, rollout_worker, evaluator, writer):
    """Train DDPG with multiple workers"""
    n_epochs = int(PARAMS['num_timesteps'] // PARAMS['n_cycles'] // PARAMS['T'] // PARAMS['num_workers'])
    for epoch in range(n_epochs):
        print('Epoch {} of {} epochs'.format(epoch, n_epochs))
        # perform rollout and update the target network
        for _ in range(PARAMS['n_cycles']):
            episode = rollout_worker.generate_rollouts()
            policy.store_episode(episode)
            for _ in range(PARAMS['n_batches']):
                policy.train()
            policy.update_target_net()
        # test
        test_scores = []
        for _ in range(PARAMS['n_test_rollouts']):
            evaluator.generate_rollouts()
            test_scores.append(evaluator.mean_success)
        writer.add_scalar('score', np.mean(test_scores), epoch)
        print('\tEpoch {}: {}'.format(epoch, np.mean(test_scores)))

        # make sure that different threads have different seeds
        MPI.COMM_WORLD.Bcast(np.random.uniform(size=(1,)), root=0)


def main():
    choose_gpu()
    args = parse_args()
    seed = set_seed(args.seed)
    env = make_vec_env(args.env, 'robotics', args.num_workers, seed=seed,
                       reward_scale=1.0, flatten_dict_observations=False)
    env.get_images()
    seed = set_seed(args.seed)
    get_dims(env)
    PARAMS['sample_her_transitions'] = make_sample_her_transitions(PARAMS['distance_threshold'])
    PARAMS['log_dir'] = 'runs/env=%s_seed=%s' % (args.env, seed)
    shutil.rmtree(PARAMS['log_dir'], ignore_errors=True)
    print('logging to:', PARAMS['log_dir'])
    writer = SummaryWriter(PARAMS['log_dir'])

    policy = DDPG(PARAMS)
    rollout_worker = RolloutWorker(env, policy, PARAMS)
    evaluator = RolloutWorker(env, policy, PARAMS, evaluate=True)
    train(policy, rollout_worker, evaluator, writer)


if __name__ == '__main__':
    main()
