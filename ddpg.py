from actor_critic import ActorCritic
from collections import OrderedDict
from normalizer import Normalizer
from replay_buffer import ReplayBuffer

import copy
import numpy as np
import torch
import torch.optim as optim


class DDPG(object):
    def __init__(self, params):
        """Implementation of DDPG that is used in combination with Hindsight Experience Replay (HER).

        Args:
            input_dims (dict of ints): dimensions for the observation (o), the goal (g), and the
                actions (u)
            buffer_size (int): number of transitions that are stored in the replay buffer
            polyak (float): coefficient for Polyak-averaging of the target network
            batch_size (int): batch size for training
            Q_lr (float): learning rate for the Q (critic) network
            pi_lr (float): learning rate for the pi (actor) network
            norm_eps (float): a small value used in the normalizer to avoid numerical instabilities
            norm_clip (float): normalized inputs are clipped to be in [-norm_clip, norm_clip]
            clip_obs (float): clip observations before normalization to be in [-clip_obs, clip_obs]
            T (int): the time horizon for rollouts
            rollout_batch_size (int): number of parallel rollouts per DDPG agent
            clip_return (float): clip returns to be in [-clip_return, clip_return]
            sample_transitions (function) function that samples from the replay buffer
            gamma (float): gamma used for Q learning updates
            reuse (boolean): whether or not the networks should be reused
        """

        self.input_dims = params['dims']
        self.buffer_size = params['buffer_size']
        self.polyak = params['polyak']
        self.batch_size = params['batch_size']
        self.Q_lr = params['lr']
        self.pi_lr = params['lr']
        self.norm_eps = params['norm_eps']
        self.norm_clip = params['norm_clip']
        self.clip_obs = params['clip_obs']
        self.T = params['T']
        self.rollout_batch_size = params['num_workers']
        self.clip_return = params['clip_return']
        self.sample_transitions = params['sample_her_transitions']
        self.gamma = params['gamma']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.replay_strategy = params['replay_strategy']
        if self.replay_strategy == 'future':
            self.use_goal = True
        else:
            self.use_goal = False

        self.dimo = self.input_dims['o']
        self.dimg = self.input_dims['g']
        self.dimu = self.input_dims['u']

        stage_shapes = OrderedDict()
        for key in sorted(self.input_dims.keys()):
            if key.startswith('info_'):
                continue
            stage_shapes[key] = (None, self.input_dims[key])
        stage_shapes['o_2'] = stage_shapes['o']
        stage_shapes['r'] = (None,)
        self.stage_shapes = stage_shapes

        self.create_network()

        # Configure the replay buffer.
        buffer_shapes = {key: (self.T-1 if key != 'o' else self.T, self.input_dims[key])
                         for key, val in self.input_dims.items()}
        buffer_shapes['g'] = (buffer_shapes['g'][0], self.dimg)
        buffer_shapes['ag'] = (self.T, self.dimg)

        buffer_size = (self.buffer_size // self.rollout_batch_size) * self.rollout_batch_size
        self.buffer = ReplayBuffer(buffer_shapes, buffer_size, self.T, self.sample_transitions)

    def random_action(self, n):
        return torch.tensor(np.random.uniform(low=-1., high=1., size=(n, self.dimu)).astype(np.float32))

    def get_actions(self, o, g, noise_eps=0., random_eps=0.):
        actions = self.main.get_action(o, g)

        noise = (noise_eps * np.random.randn(actions.shape[0], 4)).astype(np.float32)
        actions += torch.tensor(noise).to(self.device)

        actions = torch.clamp(actions, -1., 1.)
        eps_greedy_noise = np.random.binomial(1, random_eps, actions.shape[0]).reshape(-1, 1)
        random_action = self.random_action(actions.shape[0]).to(self.device)
        actions += torch.tensor(eps_greedy_noise.astype(np.float32)).to(self.device) * (
                    random_action - actions)  # eps-greedy
        return actions

    def store_episode(self, episode_batch):
        """
        episode_batch: array of batch_size x (T or T+1) x dim_key
                       'o' is of size T+1, others are of size T
        """
        self.buffer.store_episode(episode_batch)

        # add transitions to normalizer
        episode_batch['o_2'] = episode_batch['o'][:, 1:, :]
        episode_batch['ag_2'] = episode_batch['ag'][:, 1:, :]
        shape = episode_batch['u'].shape
        num_normalizing_transitions = shape[0] * shape[1]  # num_rollouts * (rollout_horizon - 1) --> total steps per cycle
        transitions = self.sample_transitions(episode_batch, num_normalizing_transitions)

        self.o_stats.update(transitions['o'])
        self.o_stats.recompute_stats()

        if self.use_goal:
            self.g_stats.update(transitions['g'])
            self.g_stats.recompute_stats()

    def sample_batch(self):
        transitions = self.buffer.sample(self.batch_size)
        return [transitions[key] for key in self.stage_shapes.keys()]

    def train(self):
        batch = self.sample_batch()
        batch_dict = OrderedDict([(key, batch[i].astype(np.float32).copy())
                                 for i, key in enumerate(self.stage_shapes.keys())])
        batch_dict['r'] = np.reshape(batch_dict['r'], [-1, 1])

        main_batch = batch_dict
        target_batch = batch_dict.copy()
        target_batch['o'] = batch_dict['o_2']

        self.main.compute_all(main_batch['o'], main_batch['g'],
                              main_batch['u'])
        self.target.compute_all(target_batch['o'], target_batch['g'],
                                target_batch['u'])

        # Q function loss
        rewards = torch.tensor(main_batch['r'].astype(np.float32)).to(self.device)
        discounted_reward = self.gamma * self.target.q_pi
        target = torch.clamp(rewards + discounted_reward, -self.clip_return, 0.)
        q_loss = torch.nn.MSELoss()(target.detach(), self.main.q)

        self.critic_optimizer.zero_grad()
        q_loss.backward()
        self.critic_optimizer.step()

        # policy loss
        pi_loss = -self.main.q_pi.mean()
        pi_loss += (self.main.pi ** 2).mean()

        self.actor_optimizer.zero_grad()
        pi_loss.backward()
        self.actor_optimizer.step()

    def update_target_net(self):
        beta = 1. - self.polyak
        for target, source in zip(self.target.parameters(), self.main.parameters()):
            target.data.copy_(beta * source.data + self.polyak * target.data)

    def create_network(self):
        # for actor network
        self.o_stats = Normalizer(size=self.dimo, eps=self.norm_eps, default_clip_range=self.norm_clip)
        if self.use_goal:
            self.g_stats = Normalizer(size=self.dimg, eps=self.norm_eps, default_clip_range=self.norm_clip)
        else:
            self.g_stats = None

        self.main = ActorCritic(self.o_stats, self.g_stats, self.input_dims, self.use_goal).to(self.device)
        self.target = ActorCritic(self.o_stats, self.g_stats, self.input_dims, self.use_goal).to(self.device)
        self.target.actor = copy.deepcopy(self.main.actor)
        self.target.critic = copy.deepcopy(self.main.critic)

        self.actor_optimizer = optim.Adam(self.main.actor.parameters(), lr=self.pi_lr)
        self.critic_optimizer = optim.Adam(self.main.critic.parameters(), lr=self.Q_lr)
