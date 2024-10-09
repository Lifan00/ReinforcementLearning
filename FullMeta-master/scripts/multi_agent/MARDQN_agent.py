import os
import torch
from tqdm import trange
from hvac.entity.EnvCore import EnvCore
from hvac.entity.environment import Env
from hvac.tools.config import DICT_PATH
from hvac.tools.scripts import DataRecorder, PmvBasedController, PklRecorder
from rl_algorithm.multi_algorithm.MARDQN_net import RDQN
import json

def train():
    EPISODES = 20

    env = Env()
    NUM_AGENT = env.n_agents
    print("NUM_AGENT:",NUM_AGENT)
    file =open(DICT_PATH, 'r', encoding='utf-8')
    action_dict = json.load(file)
    agents=[]
    for i in range(NUM_AGENT):
        agent = RDQN()
        agents.append(agent)

    with trange(EPISODES) as tqdm_episodes:
        for i, _ in enumerate(tqdm_episodes):
            state = env.reset()
            done = [False] * NUM_AGENT
            counter = 0

            while not any(done):
                if counter % 15 == 0:
                    multi_action = []
                    action = []
                    for agent_i in range(NUM_AGENT):
                        agent = agents[agent_i]
                        single_obs = state[agent_i]
                        single_action=agent.network.select_action(torch.FloatTensor(single_obs).unsqueeze(0))
                        multi_action.append(single_action)
                        true_action=action_dict[f"{single_action}"]
                        action.append(true_action)
                    next_state, reward, done, info = env.step(action)
                    if any(done):
                        break
                    for agent_i in range(NUM_AGENT):
                        agents[agent_i].learn(state[agent_i], multi_action[agent_i], reward[agent_i], next_state[agent_i], done[agent_i])
                else:
                    next_state, reward, done, info = env.step(action)
                state = next_state
                counter += 1

if __name__ == "__main__":
    os.chdir("../../")
    train()