import pickle
import random
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


def discount_cumsum(x, gamma):
    disc_cumsum = np.zeros_like(x)
    disc_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        disc_cumsum[t] = x[t] + gamma * disc_cumsum[t + 1]
    return disc_cumsum


class D4RLTrajectoryDataset(Dataset):
    def __init__(self, dataset_path, context_len, rtg_scale):

        self.context_len = context_len

        # load dataset
        with open(dataset_path, 'rb') as f:
            self.trajectories = pickle.load(f)

        # calculate min len of traj, state mean and variance
        # and returns_to_go for all traj
        min_len = 10 ** 6
        states = []
        for traj in self.trajectories:
            traj_len = traj['observations'].shape[0]
            min_len = min(min_len, traj_len)
            states.append(traj['observations'])
            # calculate returns to go and rescale them
            traj['returns_to_go'] = discount_cumsum(traj['rewards'], 1.0) / rtg_scale

        # used for input normalization
        states = np.concatenate(states, axis=0)
        self.state_mean, self.state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

        # normalize states
        for traj in self.trajectories:
            traj['observations'] = (traj['observations'] - self.state_mean) / self.state_std

    def get_state_stats(self):
        return self.state_mean, self.state_std

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        traj_len = traj['observations'].shape[0]

        if traj_len >= self.context_len:
            # sample random index to slice trajectory
            si = random.randint(0, traj_len - self.context_len)

            states = torch.from_numpy(traj['observations'][si: si + self.context_len])
            actions = torch.from_numpy(traj['actions'][si: si + self.context_len])
            returns_to_go = torch.from_numpy(traj['returns_to_go'][si: si + self.context_len])
            timesteps = torch.arange(start=si, end=si + self.context_len, step=1)

            # all ones since no padding
            traj_mask = torch.ones(self.context_len, dtype=torch.long)

        else:
            padding_len = self.context_len - traj_len

            # padding with zeros
            states = torch.from_numpy(traj['observations'])
            states = torch.cat([states,
                                torch.zeros(([padding_len] + list(states.shape[1:])),
                                            dtype=states.dtype)],
                               dim=0)

            actions = torch.from_numpy(traj['actions'])
            actions = torch.cat([actions,
                                 torch.zeros(([padding_len] + list(actions.shape[1:])),
                                             dtype=actions.dtype)],
                                dim=0)

            returns_to_go = torch.from_numpy(traj['returns_to_go'])
            returns_to_go = torch.cat([returns_to_go,
                                       torch.zeros(([padding_len] + list(returns_to_go.shape[1:])),
                                                   dtype=returns_to_go.dtype)],
                                      dim=0)

            timesteps = torch.arange(start=0, end=self.context_len, step=1)

            traj_mask = torch.cat([torch.ones(traj_len, dtype=torch.long),
                                   torch.zeros(padding_len, dtype=torch.long)],
                                  dim=0)

        return timesteps, states, actions, returns_to_go, traj_mask


start_time = datetime.now().replace(microsecond=0)
start_time_str = start_time.strftime("%y-%m-%d-%H-%M-%S")

env_name = 'HalfCheetah-v3'
# dataset_path = r'./halfcheetah-medium-v2.pkl'
dataset_path = r'./dataset.pkl'
device = torch.device('cuda')
context_len = 20
rtg_scale = 1000
batch_size = 64
n_blocks = 3
embed_dim = 128
n_heads = 1
dropout_p = 0.1
lr = 1e-4
wt_decay = 1e-4
warmup_steps = 10000

print("=" * 60)
print("start time: " + start_time_str)
print("=" * 60)

print("device set to: " + str(device))
print("dataset path: " + dataset_path)

traj_dataset = D4RLTrajectoryDataset(dataset_path, context_len, rtg_scale)

traj_data_loader = DataLoader(traj_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)

data_iter = iter(traj_data_loader)

# env = gym.make(env_name)
# state_dim = env.observation_space.shape[0]
# act_dim = env.action_space.shape[0]
# print(env_name + " state_space:\t" + str(state_dim))
# print(env_name + " action_space:\t" + str(act_dim))

# model = DecisionTransformer(
#     state_dim=state_dim,
#     act_dim=act_dim,
#     n_blocks=n_blocks,
#     h_dim=embed_dim,
#     context_len=context_len,
#     n_heads=n_heads,
#     drop_p=dropout_p,
# ).to(device)
#
# optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wt_decay)
#
# scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda steps: min((steps + 1) / warmup_steps, 1))

print("=" * 60)
try:
    timesteps, states, actions, returns_to_go, traj_mask = next(data_iter)
except StopIteration:
    data_iter = iter(traj_data_loader)
    timesteps, states, actions, returns_to_go, traj_mask = next(data_iter)

print(actions.shape)
print(states.shape)
print(timesteps.shape)
print(timesteps[0])
print(returns_to_go.shape)
print(returns_to_go[0])
print(traj_mask.shape)
print(traj_mask[0])

print(timesteps[0])
print(actions[0].shape)
print(actions[0][0])
print("k")

timesteps, states, actions, returns_to_go, traj_mask = traj_dataset.__getitem__(9)
print(timesteps)
print(actions)
print(actions.shape)

print("=" * 60)

print(timesteps.shape)
print(timesteps[0])
