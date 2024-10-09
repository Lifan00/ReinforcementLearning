import json
import os

import torch
from tqdm import trange

from hvac.entity.EnvCore import EnvCore
from hvac.tools.config import DICT_PATH
from hvac.tools.scripts import DataRecorder, PmvBasedController, PklRecorder
from rl_algorithm.rl_one.DDPG_net import DDPG

def train():
    EPISODES = 100
    ON_CONTROLLER = False
    pklEpoch = 64
    env = EnvCore()

    file = open(DICT_PATH, 'r', encoding='utf-8')
    action_dict = json.load(file)
    name = "test_perfect_data_v2"

    state_dim=env.state_space
    action_dim=len(action_dict)
    batch_size=32
    agent = DDPG(state_dim,action_dim,batch_size)
    dr = DataRecorder(name)
    pr = PklRecorder(name)
    pbc = PmvBasedController()

    with trange(EPISODES) as tqdm_episodes:
        for i, _ in enumerate(tqdm_episodes):
            dr.start(True or i == EPISODES - 1)
            pr.start(i >= EPISODES - pklEpoch)

            # noinspection PyRedeclaration
            state = env.reset()
            done = False
            counter = 0

            while not done:
                if ON_CONTROLLER:
                    if counter % 15 == 0:
                        action = pbc.select_action(state, env.get_info()[0]["ts"][0])
                        next_state, reward, done, info = env.step(action)
                        if done:  # 小小的bug
                            break
                        dr.collect(env.get_info()[0], reward)
                        pr.collect(state, action, reward, done, next_state)
                    else:
                        next_state, reward, done, info = env.step(action)
                else:
                    if counter % 15 == 0:
                        action = agent.select_action(torch.FloatTensor(state))
                        action_index = torch.argmax(action, dim=0)
                        choose_action = action_index.item()
                        true_action = action_dict[f"{choose_action}"]
                        next_state, reward, done, info = env.step(true_action)
                        if done:
                            break
                        if counter>batch_size:
                            agent.learn()
                        dr.collect(env.get_info()[0], reward)
                        pr.collect(state, true_action, reward, done, next_state)
                    else:
                        next_state, reward, done, info = env.step(true_action)
                    agent.memory.push(state,action,reward,next_state,done)
                state = next_state
                counter += 1
            dr.end()
            pr.end()
            # dr.print()
    # pr.save()
    agent.save_loss()
    print("Training finished!")

if __name__ == "__main__":
    os.chdir("../../")
    train()
