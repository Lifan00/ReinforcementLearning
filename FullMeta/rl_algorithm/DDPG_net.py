import torch.autograd
import torch.optim as optim
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd
import numpy as np
import gym
from collections import deque
import random


class OUNoise(object):
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_space.shape[0]
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return action + ou_state


class NormalizedEnv(gym.ActionWrapper):
    def _action(self, action):
        act_k = 1. / 2.
        act_b = 3 / 2.
        return act_k * action + act_b

    def _reverse_action(self, action):
        act_k_inv = 2. / 1.
        act_b = 3 / 2.
        return act_k_inv * (action - act_b)


class Memory:
    def __init__(self, max_size,batch_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.batch_size=batch_size

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, np.array([reward]), next_state, done)
        self.buffer.append(experience)

    def sample(self):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = random.sample(self.buffer, self.batch_size)

        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(np.array(state))
            action_batch.append(np.array(action))
            reward_batch.append(np.array(reward))
            next_state_batch.append(np.array(next_state))
            done_batch.append(np.array(done))

        state_batch = np.array(state_batch)
        action_batch = np.array(action_batch)
        reward_batch = np.array(reward_batch)
        next_state_batch = np.array(next_state_batch)
        done_batch = np.array(done_batch)

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def __len__(self):
        return len(self.buffer)


class Critic(nn.Module):
    def __init__(self, input_size, hidden_size, name, chkpt_dir, best_chkpt_dir):
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

        self.name = name
        self.chkpt_dir = chkpt_dir
        self.chkpt_file = os.path.join(chkpt_dir, self.name)
        self.best_checkpoint_dir = best_chkpt_dir
        self.best_checkpoint_file = os.path.join(self.best_checkpoint_dir, name + '_ddpg')

    def forward(self, state, action):
        """
        Params state and actions are torch tensors
        """
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        return x

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        torch.load(self.chkpt_file)

    def save_best_checkpoint(self):
        # print('... saving checkpoint ...')
        torch.save(self.state_dict(), self.best_checkpoint_file)

    def load_best_checkpoint(self):
        torch.load(self.best_checkpoint_file)



class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, name, chkpt_dir, best_chkpt_dir):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

        self.name = name
        self.chkpt_dir = chkpt_dir
        self.chkpt_file = os.path.join(chkpt_dir, self.name)

        self.best_checkpoint_dir = best_chkpt_dir
        self.best_checkpoint_file = os.path.join(self.best_checkpoint_dir, name + '_ddpg')

    def forward(self, state):
        """
        Param state is a torch tensor
        """
        # pu.db
        # print(state)
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))
        # action_index = torch.argmax(x, dim=1)
        # i = action_index.item()
        return x

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        torch.load(self.chkpt_file)

    def save_best_checkpoint(self):
        # print('... saving checkpoint ...')
        torch.save(self.state_dict(), self.best_checkpoint_file)

    def load_best_checkpoint(self):
        torch.load(self.best_checkpoint_file)

class DDPG:
    def __init__(self,num_states,num_actions,batch_size,chkpt_dir="", best_chkpt_dir="", hidden_size=256, actor_learning_rate=1e-3, critic_learning_rate=1e-3, gamma=0.99,
                 tau=1e-2, max_memory_size=50000):
        # Params
        self.num_states = num_states
        self.num_actions = num_actions
        self.gamma = gamma
        self.tau = tau
        self.chkpt_dir = chkpt_dir
        self.batch_size=batch_size

        # Networks
        self.actor = Actor(self.num_states, hidden_size, self.num_actions, name="actor", chkpt_dir=self.chkpt_dir,best_chkpt_dir=best_chkpt_dir)
        self.actor_target = Actor(self.num_states, hidden_size, self.num_actions, name="actor_target",
                                  chkpt_dir=self.chkpt_dir, best_chkpt_dir=best_chkpt_dir)
        self.critic = Critic(self.num_states+self.num_actions, hidden_size, name="critic",chkpt_dir=self.chkpt_dir, best_chkpt_dir=best_chkpt_dir)
        self.critic_target = Critic(self.num_states+self.num_actions, hidden_size,
                                    name="critic_target", chkpt_dir=self.chkpt_dir, best_chkpt_dir=best_chkpt_dir)

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        # Training
        self.memory = Memory(max_memory_size,self.batch_size)
        self.critic_criterion = nn.MSELoss()
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)

    def select_action(self, state):
        state = torch.Tensor(state)
        # state=torch.from_numpy(state)
        action = self.actor.forward(state)
        action = action.detach()
        # return action

        return action

    def learn(self):
        states, actions, rewards, next_states, _ = self.memory.sample()
        states = torch.Tensor(states)
        actions = torch.Tensor(actions)
        rewards = torch.Tensor(rewards)
        next_states = torch.Tensor(next_states)

        # Critic loss
        Qvals = self.critic.forward(states, actions)
        next_actions = self.actor_target.forward(next_states)
        next_Q = self.critic_target.forward(next_states, next_actions.detach())
        Qprime = rewards + self.gamma * next_Q
        critic_loss = self.critic_criterion(Qvals, Qprime)

        # Actor loss
        policy_loss = -self.critic.forward(states, self.actor.forward(states)).mean()

        # update networks
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # update target networks
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

    def get_model(self):
        return self.actor

    def save_models(self):
        self.actor.save_checkpoint()
        self.actor_target.save_checkpoint()
        self.critic.save_checkpoint()
        self.critic_target.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.actor_target.load_checkpoint()
        self.critic.load_checkpoint()
        self.critic_target.load_checkpoint()

    def save_best_models(self):
        self.actor.save_best_checkpoint()
        self.actor_target.save_best_checkpoint()
        self.critic.save_best_checkpoint()
        self.critic_target.save_best_checkpoint()