import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = T.device("cuda:0" if T.cuda.is_available() else "cpu")


class ReplayBuffer:
    def __init__(self, state_dim, max_size):
        self.mem_size = max_size
        self.batch_size = 256
        self.mem_cnt = 0
        self.state_memory = np.zeros((self.mem_size, state_dim))
        self.action_memory = np.zeros((self.mem_size,))
        self.reward_memory = np.zeros((self.mem_size,))
        self.next_state_memory = np.zeros((self.mem_size, state_dim))
        self.terminal_memory = np.zeros((self.mem_size,), dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done):
        mem_idx = self.mem_cnt % self.mem_size
        self.state_memory[mem_idx] = state
        self.action_memory[mem_idx] = action
        self.reward_memory[mem_idx] = reward
        self.next_state_memory[mem_idx] = state_
        self.terminal_memory[mem_idx] = done
        self.mem_cnt += 1

    def sample_buffer(self):
        mem_len = min(self.mem_size, self.mem_cnt)
        batch = np.random.choice(mem_len, self.batch_size, replace=False)
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.next_state_memory[batch]
        terminals = self.terminal_memory[batch]
        return states, actions, rewards, states_, terminals

    def ready(self):
        return self.mem_cnt > self.batch_size


class DeepQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DeepQNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, action_dim),
        )
        self.optimizer = optim.Adam(self.parameters(), lr=0.0003)
        self.to(device)

    def forward(self, state):
        return self.net(state)


class DDQN:
    def __init__(self, state_dim, action_dim, tau=0.005, eps_end=0.01, eps_dec=5e-7, max_size=1000000):
        self.gamma = 0.99
        self.tau = tau
        self.epsilon = 0.1
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.action_space = [i for i in range(action_dim)]
        self.q_eval = DeepQNetwork(state_dim=state_dim, action_dim=action_dim)
        self.q_target = DeepQNetwork(state_dim=state_dim, action_dim=action_dim)
        self.memory = ReplayBuffer(state_dim=state_dim, max_size=max_size)
        self.update_network_parameters(tau=1.0)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
        for q_target_params, q_eval_params in zip(self.q_target.parameters(), self.q_eval.parameters()):
            q_target_params.data.copy_(tau * q_eval_params + (1 - tau) * q_target_params)

    def remember(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def choose_action(self, state, isTrain=True):
        state = T.as_tensor(state, dtype=T.float32).to(device)  # 转为tensor格式
        actions = self.q_eval.forward(state)  # 经过全连接层转换为各种动作的得分
        action = T.argmax(actions).item()  # 最大得分的动作
        if (np.random.random() < self.epsilon) and isTrain:
            action = np.random.choice(self.action_space)
        return action

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
            if self.epsilon > self.eps_min else self.eps_min

    def learn(self):
        if not self.memory.ready():  # 有足够的batch再更新参数
            return
        states, actions, rewards, next_states, terminals = self.memory.sample_buffer()
        batch_idx = np.arange(256)
        # 转换为tensor
        states_tensor = T.tensor(states, dtype=T.float).to(device)
        rewards_tensor = T.tensor(rewards, dtype=T.float).to(device)
        next_states_tensor = T.tensor(next_states, dtype=T.float).to(device)
        terminals_tensor = T.tensor(terminals).to(device)
        with T.no_grad():
            q_ = self.q_eval.forward(next_states_tensor)
            next_actions = T.argmax(q_, dim=-1)
            q_ = self.q_target.forward(next_states_tensor)
            q_[terminals_tensor] = 0.0
            target = rewards_tensor + self.gamma * q_[batch_idx, next_actions]
        q = self.q_eval.forward(states_tensor)[batch_idx, actions]
        # 损失值的计算
        loss = F.mse_loss(q, target.detach())
        self.q_eval.optimizer.zero_grad()
        loss.backward()
        self.q_eval.optimizer.step()
        # 更新网络参数
        self.update_network_parameters()
        self.decrement_epsilon()
