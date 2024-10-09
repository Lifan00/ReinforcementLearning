# show_pkl.py

import pickle

# path = r'./rainbow_dqn_dataset-128epoch-newAC.pkl'
path = r'./perfect_v3/test_perfect_data_v3_20231114153426.pkl'
# path = r'./hopper-expert-v2.pkl'

f = open(path, 'rb')
trajectories = pickle.load(f)

print(trajectories[0].keys())
print(type(trajectories))
print(len(trajectories))
print(trajectories[0]['observations'].shape)
print("trajectories")
print(trajectories[0]['observations'].shape[0])
print(trajectories[0]['actions'].shape)
print(trajectories[0]['actions'])
print(trajectories[0]['actions'].shape[0])
print(trajectories[0]['actions'][0])
print(trajectories[0]['actions'][0])
