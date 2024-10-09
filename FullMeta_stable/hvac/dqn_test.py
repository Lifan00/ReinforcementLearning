# '''
# agent = Agent()
#
#
# for i_episode in range(TOTAL_EPISODES):
#     state = env.reset()
#     for i in range(0, MAX_LEN):
#         if i_episode % 10 == 0:
#             env.render(mode="human")
#         """
#         if random.random() < EPSILON:
#             action = random.randrange(0,ACTIONS_SIZE)
#         else:
#             action = agent.network.select_action(torch.FloatTensor(state).unsqueeze(0))
#         """
#         action = agent.network.select_action(torch.FloatTensor(state).unsqueeze(0))
#         next_state, reward, done, info = env.step(action)
#         agent.learn(state, action, reward, next_state, done)
#         state = next_state
#         if done:
#             break
#     if EPSILON > FINAL_EPSILON:
#         EPSILON -= (START_EPSILON - FINAL_EPSILON) / EXPLORE
#
#     # TEST
#     if i_episode % TEST_FREQUENCY == 0:
#         total_reward = 0
#         for j in range(0, AVERAGE):
#             state = env.reset()
#             for i in range(0, MAX_LEN):
#                 action = agent.network.select_action(torch.FloatTensor(state).unsqueeze(0))
#                 next_state, reward, done, info = env.step(action)
#
#                 total_reward += reward
#
#                 state = next_state
#                 if done:
#                     break
#         print('{},{:4f}'.format(i_episode, total_reward / AVERAGE))
#
# env.close()
# '''
import torch
from tqdm import trange

from rainbowdqn.rainbowdqn import Agent

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

if __name__ == "__main__":

    #####################  hyper parameters  ####################
    TOTAL_EPISODES = 30

    agent = Agent()
    for _ in trange(TOTAL_EPISODES):
        state = []
        reward = 0
        state_ = []
        action_value = agent.network.select_action(torch.FloatTensor(state).unsqueeze(0).to(device))

        agent.learn(state, action_value, reward, state_, False)
