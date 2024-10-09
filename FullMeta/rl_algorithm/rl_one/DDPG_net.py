import csv
from datetime import datetime
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
import torch as T

EPSILON = 0.9
device = T.device("cuda:0" if T.cuda.is_available() else "cpu")

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
    def __init__(self, input_size, hidden_size, name, chkpt_dir):
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

        self.name = name
        self.chkpt_dir = chkpt_dir
        self.chkpt_file = os.path.join(chkpt_dir, self.name)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        return x

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        torch.load(self.chkpt_file)

class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, name, chkpt_dir):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

        self.name = name
        self.chkpt_dir = chkpt_dir
        self.chkpt_file = os.path.join(chkpt_dir, self.name)

    def forward(self, state):
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


class DDPG:
    def __init__(self,num_states,num_actions,batch_size,chkpt_dir="", hidden_size=256, actor_learning_rate=1e-3, critic_learning_rate=1e-3, gamma=0.99,
                 tau=1e-2, max_memory_size=50000):
        # Params
        self.num_states = num_states
        self.num_actions = num_actions
        self.gamma = gamma
        self.tau = tau
        self.chkpt_dir = chkpt_dir
        self.batch_size=batch_size

        # Networks
        self.actor = Actor(self.num_states, hidden_size, self.num_actions, name="actor", chkpt_dir=self.chkpt_dir).to(device)
        self.actor_target = Actor(self.num_states, hidden_size, self.num_actions, name="actor_target",chkpt_dir=self.chkpt_dir).to(device)
        self.critic = Critic(self.num_states+self.num_actions, hidden_size, name="critic",chkpt_dir=self.chkpt_dir).to(device)
        self.critic_target = Critic(self.num_states+self.num_actions, hidden_size,name="critic_target", chkpt_dir=self.chkpt_dir).to(device)

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        # Training
        self.memory = Memory(max_memory_size,self.batch_size)
        self.critic_criterion = nn.MSELoss()
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)

        self.num_step=0
        self.critic_loss_list = []
        self.actor_loss_list = []
    # def select_action(self, state):
    #     state = T.as_tensor(state, dtype=T.float32).to(device)
    #     action = self.actor.forward(state)
    #     action = action.detach().cpu().numpy()
    #     return action
    def select_action(self, state):
        state = T.as_tensor(state, dtype=T.float32).to(device)
        if np.random.uniform() < EPSILON:
            action = self.actor.forward(state)
            action = action.detach().cpu()
        else:
            action = np.random.uniform(-1, 1, size=self.num_actions)
            action = T.as_tensor(action, dtype=T.float32)
        return action

    def learn(self):
        states, actions, rewards, next_states, _ = self.memory.sample()
        states = torch.Tensor(states).to(device)
        actions = torch.Tensor(actions).to(device)
        rewards = torch.Tensor(rewards).to(device)
        next_states = torch.Tensor(next_states).to(device)

        # Critic loss
        Qvals = self.critic.forward(states, actions)
        next_actions = self.actor_target.forward(next_states)
        next_Q = self.critic_target.forward(next_states, next_actions.detach())
        Qprime = rewards + self.gamma * next_Q
        critic_loss = self.critic_criterion(Qvals, Qprime)

        # Actor loss
        policy_loss = -self.critic.forward(states, self.actor.forward(states)).mean()

        # print("critic_loss :", critic_loss.item(), "policy_loss :", policy_loss.item())
        if self.num_step % 1000 == 0:
            self.critic_loss_list.append(critic_loss.item())
            self.actor_loss_list.append(policy_loss.item())
            # print("critic_loss :", critic_loss.item(), "policy_loss :", policy_loss.item())

        # update networks
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.num_step += 1
        # update target networks
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

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

    def save_loss(self):
        dir_path = r"./outputs/"
        os.makedirs(dir_path, exist_ok=True)
        path = os.path.join(dir_path, "actor_critic_loss_list_" + datetime.now().strftime("%Y%m%d%H%M%S") + ".csv")
        with open(path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            header = [
                "critic_loss",
                "actor_loss"
            ]
            writer.writerow(header)
            for i in range(len(self.critic_loss_list)):
                row = [
                    self.critic_loss_list[i],
                    self.actor_loss_list[i]
                ]
                writer.writerow(row)

if __name__ == '__main__':
    env = gym.make('Pendulum-v1')
    ddpg = DDPG(num_states=env.observation_space.shape[0], num_actions=env.action_space.shape[0], batch_size=64)
    for i_episode in range(1000):
        state = env.reset()[0]
        SUM_REWARD = 0
        for t in range(500):
            action = ddpg.select_action(state)
            next_state, reward, done, _,_= env.step(action)
            ddpg.memory.push(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
            SUM_REWARD += reward
            if len(ddpg.memory) > ddpg.batch_size:
                ddpg.learn()
            # env.render()
        if i_episode % 10 == 0:
            print("Episode: {}, Score: {}".format(i_episode, SUM_REWARD))
