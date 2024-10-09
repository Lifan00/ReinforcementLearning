import json

from hvac.entity.EnvCore import EnvCore
from rl_algorithm.rl_one.DQN_net import DQN

# 超参数
BATCH_SIZE = 32  # 样本数量
LR = 0.01  # 学习率
EPSILON = 0.9  # greedy policy
GAMMA = 0.9  # reward discount
TARGET_REPLACE_ITER = 100  # 目标网络更新频率
MEMORY_CAPACITY = 2000  # 记忆库容量
N_ACTIONS = 248
N_STATES = 23  # 状态维度
file = open('D:/FullMeta/hvac/data/that_dict.json', 'r', encoding='utf-8')
action_dict = json.load(file)


def dqn_main():
    env = EnvCore()
    agent = DQN()
    state = env.reset()
    for i_episode in range(20):
        env.reset()
        sum_reward = 0
        count = 0
        done = False
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action_dict[str(action)])
            agent.store_transition(state, action, reward, next_state)
            agent.learn()
            count += 1
            state = next_state
            sum_reward += reward
            if count % 10000 == 0:
                print(count)
                print(info[0]["pmv"])
        if i_episode % 2 == 0:
            print("Episode:{},Reward:{},Count:{}".format(i_episode, sum_reward, count))
        if i_episode % 64 == 0:
            agent.learn()


if __name__ == '__main__':
    dqn_main()
