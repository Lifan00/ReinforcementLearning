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
            env.reset()
            done = [False] * NUM_AGENT
            counter = 0

            while not any(done):
                if counter % 15 == 0:
                    action = []
                    for agent_i in range(NUM_AGENT):
                        true_action=action_dict[f"{1}"]
                        action.append(true_action)
                    next_state, reward, done, info = env.step(action)
                    if any(done):
                        break
                else:
                    next_state, reward, done, info = env.step(action)
                state = next_state
                counter += 1

if __name__ == "__main__":
    os.chdir("../../../")
    train()