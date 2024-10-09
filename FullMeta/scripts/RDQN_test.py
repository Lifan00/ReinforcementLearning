import os
import torch
from tqdm import trange
from hvac.entity.EnvCore import EnvCore
from hvac.tools.config import DICT_PATH
from hvac.tools.scripts import DataRecorder, PmvBasedController, PklRecorder
from rl_algorithm.rl_one.RDQN_net import RDQN
import json

def train():
    EPISODES = 164
    # EPISODES = 2
    ON_CONTROLLER = False
    pklEpoch = 64
    env = EnvCore()

    file = open(DICT_PATH, 'r', encoding='utf-8')
    action_dict = json.load(file)

    name = "test_perfect_data_v2"

    agent = RDQN()
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
                        if done:  # Ð¡Ð¡µÄbug
                            break
                        dr.collect(env.get_info()[0], reward)
                        pr.collect(state, action, reward, done, next_state)
                    else:
                        next_state, reward, done, info = env.step(action)
                else:
                    if counter == 0:
                        action = agent.network.select_action(torch.FloatTensor(state).unsqueeze(0))
                        true_action = action_dict[f"{action}"]
                        next_state, reward, done, info = env.step(true_action)
                        if done:
                            break
                        agent.learn(state, action, reward, next_state, done)
                        dr.collect(env.get_info()[0], reward)
                        pr.collect(state, true_action, reward, done, next_state)
                    else:
                        next_state, reward, done, info = env.step(true_action)

                state = next_state
                counter += 1
                # if counter % 10000 == 0:
                #     print(f"Counter:{counter}")
            dr.end()
            pr.end()
            dr.print()

    pr.save()


if __name__ == "__main__":
    os.chdir("../../")
    train()
