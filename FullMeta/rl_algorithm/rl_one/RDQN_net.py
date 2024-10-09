import os
import random
import sys
import numpy as np
import torch
import torch as T
import torch.nn as nn

BATCH_SIZE = 50
LR = 0.0005
GAMMA = 0.98
MEMORY_SIZE = 65536
MEMORY_THRESHOLD = 1000
TEST_FREQUENCY = 10
UPDATE_TIME = 50
MAX_LEN = 1000
AVERAGE = 10
NSTEP = 2
SEED = 777
random.seed(SEED)
np.random.seed(SEED)
torch.random.manual_seed(SEED)
episode = 0
ACTIONS_SIZE = 248
STATE_SIZE = 23
device = torch.device("cuda:1" if (torch.cuda.is_available()) else "cpu")


class NoisyLinear(nn.Module):
    def __init__(self, indim: int, outdim: int, std_init: float = 0.3):
        super(NoisyLinear, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.Tensor(outdim, indim))
        self.weight_sigma = nn.Parameter(torch.Tensor(outdim, indim))
        self.register_buffer("weight_epsilon", torch.Tensor(outdim, indim))

        self.bias_mu = nn.Parameter(torch.Tensor(outdim))
        self.bias_sigma = nn.Parameter(torch.Tensor(outdim))
        self.register_buffer("bias_epsilon", torch.Tensor(outdim))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = self.indim ** -0.5
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / self.indim ** 0.5)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / self.outdim ** 0.5)

    def reset_noise(self):
        epsilon_in = self.scale_noise(self.indim)
        epsilon_out = self.scale_noise(self.outdim)
        # outer product
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(
            x,
            self.weight_mu + self.weight_sigma * self.weight_epsilon,
            self.bias_mu + self.bias_sigma * self.bias_epsilon,
        )

    @staticmethod
    def scale_noise(size: int) -> torch.Tensor:
        x = torch.randn(size).to(device)
        return x.sign().mul(x.abs().sqrt())


class RM:
    def __init__(self, maxlen: int = 16384, alpha: float = 0.25, beta: float = 0.7):
        self.maxlen: int = maxlen
        self.replay: np.array = np.zeros((maxlen * 2, 5), dtype=object)
        self.value: np.array = np.zeros(maxlen * 2, dtype=np.float32)
        self.alpha = alpha
        self.beta = beta
        self.ptr = maxlen
        self.l = 0

    def renew(self, s: int):
        while s > 1:
            s //= 2
            self.value[s] = self.value[2 * s] + self.value[2 * s + 1]

    def priority_from_loss(self, loss):
        return (loss + abs(loss - 1) - 0.8) ** self.alpha

    def store(self, loss: float, data: object):
        priority = self.priority_from_loss(loss)

        def set1(i: int, p: float, d: object):
            self.value[i] = p
            self.replay[i] = d

        l = self.l
        if l < self.maxlen:
            set1(2 * l + 1, priority, data)
            set1(2 * l, self.value[l], self.replay[l])
            # self.replay[l] = 0
            self.renew(2 * l)
            self.l += 1
        else:
            # substitute
            set1(self.ptr, priority, data)
            self.renew(self.ptr)
            self.ptr += 1
            if self.ptr >= 2 * l:
                self.ptr -= l

    def extract(self, n: int):
        l = self.l
        if l <= n:
            raise IndexError(f"impossible to take {n} elements in size {l}")

        def extract1() -> int:
            i = 1
            while i < l:
                param = random.uniform(0, self.value[i])
                if param < self.value[2 * i]:
                    i = 2 * i
                else:
                    i = 2 * i + 1
            return i

        sample_id = np.array([extract1() for _ in range(0, n)])
        coeff = (self.value[sample_id] * (l / self.value[1])) ** (-self.beta)
        return sample_id, coeff

    def batch_renew(self, indexes, losses):
        priorities = self.priority_from_loss(losses)
        for i in range(0, len(indexes)):
            self.value[indexes[i]] = priorities[i]
            self.renew(indexes[i])

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(STATE_SIZE, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 128)
        self.fc2_qval = nn.Linear(128, 64)
        self.q_avg = nn.Linear(64, 1)
        self.E = NoisyLinear(128, ACTIONS_SIZE)
        self.Var = nn.Linear(128, ACTIONS_SIZE)

    def forward(self, s):
        u = self.relu(self.fc1(s))
        if sys.getsizeof(u) != 80:
            print(sys.getsizeof(u))
        adv_1 = self.relu(self.fc2(u))
        Q_avg = self.q_avg(self.relu(self.fc2_qval(u)))
        Q_adv, QVar = self.E(adv_1), torch.abs(self.Var(adv_1))
        return (Q_avg - torch.mean(Q_adv, dim=1).unsqueeze(1)) + Q_adv, QVar

    def select_action(self, state):
        state = T.as_tensor(state, dtype=T.float32).to(device)
        with torch.no_grad():
            Q, _ = self.forward(state)
            action_index = torch.argmax(Q, dim=1)
            i = action_index.item()
        return i

    def reset_noise(self):
        self.E.reset_noise()

    def save_checkpoint(self,checkpoint_file):
        # print('... saving checkpoint ...')
        T.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self,checkpoint_file):
        self.load_state_dict(T.load(checkpoint_file))


class CrossEntropyLossOf2NormalDistributions(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, m1, s1, m2, s2):
        return (s1 ** 2 + (m1 - m2) ** 2) / (2 * s2 ** 2) + torch.log(s2)


class RDQN(object):
    def __init__(self,name):
        self.network, self.target_network = Net().to(device), Net().to(device)
        self.target_network.train(False)
        self.memory = RM(MEMORY_SIZE)
        self.episode = 0
        self.optimizer = torch.optim.AdamW(self.network.parameters(), lr=LR)
        self.loss_func = CrossEntropyLossOf2NormalDistributions().to(device)
        self.nstate = 0
        self.naction = 0
        self.nreward = 0
        self.nnext_state = 0
        self.ndone = 0
        self.nstep = 0
        self.GAMMA = GAMMA ** NSTEP
        self.name=name

    def compute_loss(self, state, action, reward, next_state, mask):
        actions_value, _ = self.network.forward(next_state)
        next_action = torch.unsqueeze(torch.max(actions_value, 1)[1], 1).to(device)
        evals = self.network.forward(state)
        eval_q, eval_s = evals[0].gather(1, action), evals[1].gather(1, action)
        nexts = self.target_network.forward(next_state)
        next_q, next_s = nexts[0].gather(1, next_action), nexts[1].gather(1, next_action)
        target_q = reward + self.GAMMA * next_q * mask
        target_s = self.GAMMA * next_s * mask  # + 0.0001
        losses = self.loss_func(target_q, target_s, eval_q, eval_s)
        return losses

    def learn(self, state, action, reward, next_state, done):
        if self.nstep == 0:
            self.nstate = state
            self.naction = action
            self.nreward = 0
        self.nreward += reward * (GAMMA ** self.nstep)
        self.nstep += 1

        if self.nstep >= NSTEP or done:
            self.nstep = 0
            self.nnext_state = next_state
            self.ndone = done
            with torch.no_grad():
                state = np.array(self.nstate, dtype=np.float32)
                action = np.array([self.naction], dtype=np.int64)
                reward = np.array([self.nreward], dtype=np.float32)
                next_state = np.array(self.nnext_state, dtype=np.float32)
                mask = np.array([1 - self.ndone], dtype=np.float32)

                append_loss = self.compute_loss(
                    torch.FloatTensor(state).unsqueeze(0).to(device),
                    torch.LongTensor(action).unsqueeze(0).to(device),
                    torch.FloatTensor(reward).unsqueeze(0).to(device),
                    torch.FloatTensor(next_state).unsqueeze(0).to(device),
                    torch.FloatTensor(mask).unsqueeze(0).to(device),
                )
                data = np.array([state, action, reward, next_state, mask], dtype=object)
                self.memory.store(append_loss, data)

        if self.episode % UPDATE_TIME == 0:
            self.target_network.load_state_dict(self.network.state_dict())
        self.network.reset_noise()
        self.target_network.reset_noise()
        self.episode += 1
        if self.memory.l <= MEMORY_THRESHOLD:
            return

        id, coeff = self.memory.extract(BATCH_SIZE)
        A = self.memory.replay[id]
        state = torch.FloatTensor(np.stack(A[:, 0])).to(device)
        action = torch.LongTensor(np.stack(A[:, 1])).to(device)
        reward = torch.FloatTensor(np.stack(A[:, 2])).to(device)
        next_state = torch.FloatTensor(np.stack(A[:, 3])).to(device)
        mask = torch.FloatTensor(np.stack(A[:, 4])).to(device)

        losses = self.compute_loss(state, action, reward, next_state, mask).squeeze()
        loss1 = torch.dot(torch.FloatTensor(coeff).to(device), losses)
        self.optimizer.zero_grad()
        loss1.backward()
        self.optimizer.step()
        with torch.no_grad():
            losses = self.compute_loss(state, action, reward, next_state, mask).squeeze()
            self.memory.batch_renew(id, losses)

    def save_models(self,checkpoint_dir):
        self.network.save_checkpoint(os.path.join(checkpoint_dir, self.name+"_network"))
        self.target_network.save_checkpoint(os.path.join(checkpoint_dir, self.name+"_target_network"))

    def load_models(self,checkpoint_dir):
        self.network.load_checkpoint(os.path.join(checkpoint_dir, self.name+"_network"))
        self.target_network.load_checkpoint(os.path.join(checkpoint_dir, self.name+"_target_network"))