import pickle

import numpy as np
from tqdm import tqdm

path_name = 'test_perfect_data_v2_20231123215003'
path = './' + path_name + '.pkl'
# path = r'./hopper-expert-v2.pkl'

f1 = open(path, 'rb')
trajs = pickle.load(f1)

# 35028   -9
# dict_keys(['actions', 'observations', 'next_observations', 'rewards', 'terminals'])
SIZE = 35037 // 12

list_all = []

# traj_last = trajs[127]

for traj in tqdm(trajs):
    arr1 = traj['actions']
    arr2 = traj['observations']
    arr3 = traj['next_observations']
    arr4 = traj['rewards']
    arr5 = traj['terminals']

    for i in range(12):  # default 12
        if i == 5 or i == 6 or i == 7 or i == 8 or True:
            split_dict = {}
            start_idx = i * SIZE
            end_idx = (i + 1) * SIZE
            split_dict['actions'] = arr1[start_idx:end_idx, :].astype(np.float32)
            split_dict['observations'] = arr2[start_idx:end_idx, :].astype(np.float32)
            split_dict['next_observations'] = arr3[start_idx:end_idx, :].astype(np.float32)
            split_dict['rewards'] = arr4[start_idx:end_idx].astype(np.float32)
            split_dict['terminals'] = arr5[start_idx:end_idx]
            list_all.append(split_dict)

            # split_dict = {}
            # start_idx = i * SIZE
            # end_idx = (i + 1) * SIZE
            # split_dict['actions'] = arr1[start_idx:end_idx, :]
            # split_dict['observations'] = arr2[start_idx:end_idx, :]
            # split_dict['next_observations'] = arr3[start_idx:end_idx, :]
            # split_dict['rewards'] = arr4[start_idx:end_idx]
            # split_dict['terminals'] = arr5[start_idx:end_idx]
            # list_all.append(split_dict)

        # elif i == 6 or i == 7 or i == 8 or i == 9:
        #     split_dict = {}
        #     start_idx = i * SIZE
        #     end_idx = (i + 1) * SIZE
        #     split_dict['actions'] = arr1[start_idx:end_idx, :]
        #     split_dict['observations'] = arr2[start_idx:end_idx, :]
        #     split_dict['next_observations'] = arr3[start_idx:end_idx, :]
        #     split_dict['rewards'] = arr4[start_idx:end_idx]
        #     split_dict['terminals'] = arr5[start_idx:end_idx]
        #     list_all.append(split_dict)
        #
        #     split_dict = {}
        #     start_idx = i * SIZE
        #     end_idx = (i + 1) * SIZE
        #     split_dict['actions'] = arr1[start_idx:end_idx, :]
        #     split_dict['observations'] = arr2[start_idx:end_idx, :]
        #     split_dict['next_observations'] = arr3[start_idx:end_idx, :]
        #     split_dict['rewards'] = arr4[start_idx:end_idx]
        #     split_dict['terminals'] = arr5[start_idx:end_idx]
        #     list_all.append(split_dict)

        # list_all.append(split_dict)
        # list_all.append(split_dict)
        # list_all.append(split_dict)
        # list_all.append(split_dict)
        # list_all.append(split_dict)
        # list_all.append(split_dict)
        # list_all.append(split_dict)
        # list_all.append(split_dict)
        # list_all.append(split_dict)
        # list_all.append(split_dict)
        # list_all.append(split_dict)

# print(list_all)

# pre_name = "summer_only_"
pre_name = "split"

save_name = pre_name + 'tmp' + path_name + 'v3.pkl'
with open(save_name, 'wb') as f2:
    pickle.dump(list_all, f2)
