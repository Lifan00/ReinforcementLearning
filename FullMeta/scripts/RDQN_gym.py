import os
import torch
from tqdm import trange
from rl_algorithm.rl_one.RDQN_net import RDQN
import gym

def train():
    EPISODES = 164
    env = gym.make("MountainCar-v0")
    agent = RDQN()
    with trange(EPISODES) as tqdm_episodes:
        for i, _ in enumerate(tqdm_episodes):
            # noinspection PyRedeclaration
            state,_ = env.reset()
            done = False
            counter = 0

            while not done:
                if counter % 15 == 0:
                    action = agent.network.select_action(torch.FloatTensor(state).unsqueeze(0))
                    next_state, reward, done, info,_ = env.step(action)
                    if done:
                        break
                    agent.learn(state, action, reward, next_state, done)
                else:
                    next_state, reward, done, info,_ = env.step(action)
                state = next_state
                counter += 1
                # if counter % 10000 == 0:
                #     print(f"Counter:{counter}")

if __name__ == "__main__":
    os.chdir("../../")
    train()
