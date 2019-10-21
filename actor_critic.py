import torch
import torch.nn as nn


class ActorCritic(nn.Module):
    def __init__(self, o_stats, g_stats, dims, use_goal):
        super(ActorCritic, self).__init__()
        self.actor = Actor(dims, use_goal)
        self.critic = Critic(dims, use_goal)
        self.use_goal = use_goal

        self.o_stats = o_stats
        self.g_stats = g_stats
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def compute_all(self, obs, goal, actions):
        obs = self.o_stats.normalize(obs)
        obs = torch.tensor(obs).to(self.device)
        if self.use_goal:
            goal = self.g_stats.normalize(goal)
            goal = torch.tensor(goal).to(self.device)
            policy_input = torch.cat([obs, goal], dim=1)
            policy_output = self.actor(policy_input)
            # temporary
            self.pi = policy_output
            critic_input = torch.cat([obs, goal, policy_output], dim=1)
            self.q_pi = self.critic(critic_input)
            actions = torch.tensor(actions).to(self.device)
            critic_input = torch.cat([obs, goal, actions], dim=1)
            self.q = self.critic(critic_input)
        else:
            policy_input = torch.cat([obs], dim=1)
            policy_output = self.actor(policy_input)
            # temporary
            self.pi = policy_output
            critic_input = torch.cat([obs, policy_output], dim=1)
            self.q_pi = self.critic(critic_input)
            actions = torch.tensor(actions).to(self.device)
            critic_input = torch.cat([obs, actions], dim=1)
            self.q = self.critic(critic_input)

    def get_action(self, obs, goals):
        obs = self.o_stats.normalize(obs)
        obs = torch.tensor(obs).to(self.device)
        if self.use_goal:
            goals = self.g_stats.normalize(goals)
            goals = torch.tensor(goals).to(self.device)
            policy_input = torch.cat([obs, goals], dim=1)
        else:
            policy_input = torch.cat([obs], dim=1)
        return self.actor(policy_input)

    def compute_q_values(self, obs, goals, actions):
        obs = self.o_stats.normalize(obs)
        obs = torch.tensor(obs).to(self.device)
        if self.use_goal:
            goals = self.g_stats.normalize(goals)
            goals = torch.tensor(goals).to(self.device)
            input_tensor = torch.cat([obs, goals, actions], dim=1)
        else:
            input_tensor = torch.cat([obs, actions], dim=1)
        return self.critic(input_tensor)


class Actor(nn.Module):
    def __init__(self, dims, use_goal):
        super(Actor, self).__init__()
        if use_goal:
            self.linear1 = nn.Linear(in_features=dims['o'] + dims['g'], out_features=256)
        else:
            self.linear1 = nn.Linear(in_features=dims['o'], out_features=256)
        self.linear2 = nn.Linear(in_features=256, out_features=256)
        self.linear3 = nn.Linear(in_features=256, out_features=256)
        self.linear4 = nn.Linear(in_features=256, out_features=4)
        self.actor = nn.Sequential(
            self.linear1,
            nn.ReLU(),
            self.linear2,
            nn.ReLU(),
            self.linear3,
            nn.ReLU(),
            self.linear4,
            nn.Tanh()
        )

    def forward(self, input_tensor):
        return self.actor(input_tensor)


class Critic(nn.Module):
    def __init__(self, dims, use_goal):
        super(Critic, self).__init__()
        if use_goal:
            self.linear1 = nn.Linear(in_features=dims['o'] + dims['g'] + dims['u'], out_features=256)
        else:
            self.linear1 = nn.Linear(in_features=dims['o'] + dims['u'], out_features=256)
        self.linear2 = nn.Linear(in_features=256, out_features=256)
        self.linear3 = nn.Linear(in_features=256, out_features=256)
        self.linear4 = nn.Linear(in_features=256, out_features=1)
        self.critic = nn.Sequential(
            self.linear1,
            nn.ReLU(),
            self.linear2,
            nn.ReLU(),
            self.linear3,
            nn.ReLU(),
            self.linear4,
        )

    def forward(self, input_tensor):
        return self.critic(input_tensor)
