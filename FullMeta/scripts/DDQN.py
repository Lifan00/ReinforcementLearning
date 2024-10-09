import json

from hvac.entity.EnvCore import EnvCore
from rl_algorithm.rl_one.DDQN_net import DDQN

env = EnvCore()
state_dim = env.state_space  # 23
action_space = env.action_space  # 11
max_episodes = 500
# agent 加载
agent = DDQN(state_dim=state_dim, action_dim=action_space, tau=0.005)
# all action load here
file = open('D:/FullMeta/hvac/data/that_dict.json', 'r', encoding='utf-8')
action_dict = json.load(file)  # dict


def ddqn_main():
    state = env.reset()
    for i_episode in range(20000):
        env.reset()
        sum_reward = 0
        for i_step in range(300):
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action_dict[str(action)])
            agent.memory.store_transition(state, action, reward, next_state, done)
            agent.learn()
            state = next_state
            sum_reward += reward
            if done:
                break
        if i_episode % 10 == 0:
            print("Episode:{},Reward:{}".format(i_episode, sum_reward))
        if i_episode % 64 == 0:
            agent.learn()


if __name__ == '__main__':
    ddqn_main()
