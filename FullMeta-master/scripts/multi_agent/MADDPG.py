import json
import numpy as np
import torch
import torch.nn as nn
import os
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from rl_algorithm.multi_algorithm.MADDPG_net import Agent
from hvac.entity.environment import Env
from hvac.tools.config import DICT_PATH
from hvac.tools.scripts import DataRecorder, PmvBasedController, PklRecorder

# Write a function to convert multi_obs in dict type to state in dict type
# Set device

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("Using device: ", device)

#Set parameter
NUM_EPISODE =20
LR_ACTOR = 0.001
LR_CRITIC = 0.001
HIDDEN_DIM = 256
GAMMA = 0.99
TAU = 0.01
MEMORY_SIZE = 50000
BATCH_SIZE = 32
env =Env()
NUM_AGENT = env.n_agents
print("NUM_AGENT:",NUM_AGENT)
file = open(DICT_PATH, 'r', encoding='utf-8')
action_dict = json.load(file)
# action_dict = {
#     "0": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     "1": [0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1],
#     # "2": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#     # "3": [-1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1]
# }
# 1 Initialize agents
# 1.1 Get obs_dim, state_dim
obs_dim = env.state_space
state_dim = sum(obs_dim)
# 1.2 Get action_dim
action_dim = len(action_dict)
# 1.3 init all agents
name=f"Test_Multi_agents_{NUM_AGENT}_data_v0"
agents = []
dr=[]
pr=[]
pbc=[]
for agent_i in range(NUM_AGENT):
    # print(f"Initializing agent {agent_i}")
    agent = Agent(memo_size=MEMORY_SIZE, obs_dim=obs_dim[agent_i], state_dim=state_dim, n_agent=NUM_AGENT,
                  action_dim=action_dim, alpha=LR_ACTOR, beta=LR_CRITIC, fc1_dims=HIDDEN_DIM,
                  fc2_dims=HIDDEN_DIM, gamma=GAMMA, tau=TAU, batch_size=BATCH_SIZE)
    agents.append(agent)
    dr.append(DataRecorder(name+f'_agent_{agent_i}'))
    pr.append(PklRecorder(name+f'_agent_{agent_i}'))
    pbc.append(PmvBasedController())

# Save models
scenario = "Our environment "
print(f"Scenario: {scenario}")
current_path = os.path.dirname(os.path.realpath(__file__))
agent_path = current_path + '/models/' + scenario + '/'
print("agent_path:",agent_path)
timestamp = time.strftime("%Y%m%d%H%M%S")

def multi_obs_to_state(multi_obs):
    state = np.array([])
    for agent_obs in multi_obs:
        state = np.concatenate([state, agent_obs])
    return state
class main():
    def __init__(self):
        self.train()
        # self.save()
        # self.test()

    def train(self):
        # 2 Training loop
        pklEpoch=64
        EPISODE_REWARD_BUFFER = []
        for episode_i in tqdm(range(NUM_EPISODE)):
            i_step = 0
            multi_obs= env.reset()
            episode_reward = 0
            multi_done =[False] * NUM_AGENT
            for agent_i in range(NUM_AGENT):
                dr[agent_i].start(True or episode_i == NUM_EPISODE - 1)
                pr[agent_i].start(episode_i >= NUM_EPISODE - pklEpoch)

            while not any(multi_done):
                if i_step%15==0:
                    multi_action = []
                    multi_action_excel = []
                    # 2.1 Collect actions from all agents
                    for agent_i in range(NUM_AGENT):
                        agent = agents[agent_i]
                        single_obs = multi_obs[agent_i]
                        single_action=agent.get_action(single_obs)
                        multi_action.append(single_action)
                        single_action = np.argmax(single_action) # take action based on obs
                        multi_action_excel.append(action_dict[f"{single_action}"])
                    multi_next_obs, multi_reward, multi_done, info = env.step(multi_action_excel)
                    if any(multi_done):
                        break
                    for agent_i in range(NUM_AGENT):
                        dr[agent_i].collect(env.get_info(agent_i)[0], multi_reward[agent_i],env.get_home(agent_i))
                        pr[agent_i].collect(multi_obs[agent_i], multi_action_excel, multi_reward[agent_i], multi_done[agent_i], multi_next_obs[agent_i])
                    state = multi_obs_to_state(multi_obs)
                    next_state = multi_obs_to_state(multi_next_obs)  # why the same as state?
                    # 2.3 Add memory (obs, next_obs, state, next_state, action, reward, done)
                    for agent_i in range(NUM_AGENT):
                        agent = agents[agent_i]
                        single_obs = multi_obs[agent_i]
                        single_next_obs = multi_next_obs[agent_i]
                        single_action = multi_action[agent_i]  # 5 continuous actions
                        single_reward = multi_reward[agent_i]
                        single_done = multi_done[agent_i]
                        agent.replay_buffer.add_memo(single_obs, single_next_obs, state, next_state, single_action, single_reward,single_done)
                    self.learn()
                else:
                    multi_next_obs, multi_reward, multi_done, info = env.step(multi_action_excel)
                # 2.2 Execute action at and observe reward rt and new state st+1
                i_step+=1
                multi_obs = multi_next_obs
                episode_reward += sum([single_reward for single_reward in multi_reward])/NUM_AGENT
                if i_step % 10000 == 0:
                    print("Step :", i_step)
            for agent_i in range(NUM_AGENT):
                dr[agent_i].end()
                pr[agent_i].end()
                dr[agent_i].print()
            EPISODE_REWARD_BUFFER.append(episode_reward)
        for agent_i in range(NUM_AGENT):
            pr[agent_i].save()

    def save(self):
        print(f"Saving model !")
        for agent_i in range(NUM_AGENT):
            agent = agents[agent_i]
            flag = os.path.exists(agent_path)
            if not flag:
                os.makedirs(agent_path)
            torch.save(agent.actor.state_dict(), f'{agent_path}' + f'agent_{agent_i}_actor_{scenario}_{timestamp}.pth')
        print(f"Model saved !")

    def learn(self):
        # 2.4 Update target networks every TARGET_UPDATE_INTERVAL
        # Start learning
        # Collect next actions of all agents
        multi_batch_obses = []
        multi_batch_next_obses = []
        multi_batch_states = []
        multi_batch_next_states = []
        multi_batch_actions = []
        multi_batch_next_actions = []
        multi_batch_online_actions = []
        multi_batch_rewards = []
        multi_batch_dones = []

        # 2.4.1 Sample a batch of memories
        for agent_i in range(NUM_AGENT):
            agent = agents[agent_i]
            batch_obses_tensor, batch_next_obses_tensor, batch_states_tensor, batch_next_states_tensor, \
                batch_actions_tensor, batch_rewards_tensor, batch_dones_tensor = agent.replay_buffer.sample()

            # Multiple + batch
            multi_batch_obses.append(batch_obses_tensor)
            multi_batch_next_obses.append(batch_next_obses_tensor)
            multi_batch_states.append(batch_states_tensor)
            multi_batch_next_states.append(batch_next_states_tensor)
            multi_batch_actions.append(batch_actions_tensor)

            single_batch_next_action = agent.target_actor.forward(batch_next_obses_tensor)  # a' = target(o')
            multi_batch_next_actions.append(single_batch_next_action)

            single_batch_online_action = agent.actor.forward(batch_obses_tensor)  # a = online(o)
            multi_batch_online_actions.append(single_batch_online_action)

            multi_batch_rewards.append(batch_rewards_tensor)
            multi_batch_dones.append(batch_dones_tensor)

        multi_batch_actions_tensor = torch.cat(multi_batch_actions, dim=1).to(device)
        multi_batch_next_actions_tensor = torch.cat(multi_batch_next_actions, dim=1).to(device)
        multi_batch_online_actions_tensor = torch.cat(multi_batch_online_actions, dim=1).to(device)

        # 2.4.2 Update critic and actor
        for agent_i in range(NUM_AGENT):
            agent = agents[agent_i]
            # batch_actions_tensor = multi_batch_actions[agent_i]
            # batch_obses_tensor = multi_batch_obses[agent_i]
            batch_states_tensor = multi_batch_states[agent_i]
            batch_next_states_tensor = multi_batch_next_states[agent_i]
            batch_rewards_tensor = multi_batch_rewards[agent_i]
            batch_dones_tensor = multi_batch_dones[agent_i]

            # 2.4.2.1 Calculate target Q using target critic
            critic_target_q = agent.target_critic.forward(batch_next_states_tensor,
                                                          multi_batch_next_actions_tensor.detach())
            y = (batch_rewards_tensor + (1 - batch_dones_tensor) * agent.gamma * critic_target_q).flatten()

            # 2.4.2.2 Calculate current Q using critic
            critic_q = agent.critic.forward(batch_states_tensor, multi_batch_actions_tensor).flatten()

            # 2.4.2.3 Update critic
            critic_loss = nn.MSELoss()(y, critic_q)
            agent.critic.optimizer.zero_grad()
            critic_loss.backward()
            agent.critic.optimizer.step()

            # 2.4.2.4 Update actor
            actor_loss = agent.critic.forward(batch_states_tensor,multi_batch_online_actions_tensor.detach()).flatten()
            actor_loss = -torch.mean(actor_loss)
            agent.actor.optimizer.zero_grad()
            actor_loss.backward()
            agent.actor.optimizer.step()
            # 2.4.2.5 Update target critic
            for target_param, param in zip(agent.target_critic.parameters(), agent.critic.parameters()):
                target_param.data.copy_(agent.tau * param.data + (1.0 - agent.tau) * target_param.data)
            # 2.4.2.6 Update target actor
            for target_param, param in zip(agent.target_actor.parameters(), agent.actor.parameters()):
                target_param.data.copy_(agent.tau * param.data + (1.0 - agent.tau) * target_param.data)

    def test(self):
        print("Testing!")
        EPISODE_REWARD_BUFFER=[]
        for test_epi_i in tqdm(range(10)):
            multi_obs= env.reset()
            multi_done =[False] * NUM_AGENT
            episode_reward=0
            i_step=0
            while not any(multi_done):
                multi_action = []
                multi_action_excel = []
               # 2.1 Collect actions from all agents
                for agent_i in range(NUM_AGENT):
                   agent = agents[agent_i]
                   agent.actor.load_checkpoint(f'{agent_path}' + f'agent_{agent_i}_actor_{scenario}_{timestamp}.pth')
                   single_obs = multi_obs[agent_i]
                   single_action=agent.get_action(single_obs)
                   multi_action.append(single_action)
                   single_action = np.argmax(single_action) # take action based on obs
                   multi_action_excel.append(action_dict[f"{single_action}"])
                multi_next_obs, multi_reward, multi_done, info = env.step(multi_action_excel)
                multi_obs = multi_next_obs
                i_step+=1
                if any(multi_done):
                    break
                episode_reward += sum([single_reward for single_reward in multi_reward]) / NUM_AGENT
                if i_step % 10000 == 0:
                    print("Step :", i_step)
            EPISODE_REWARD_BUFFER.append(episode_reward)
            print(f"Episode Test : {test_epi_i} Reward: {round(episode_reward, 4)}")
            print(f"Episode Reward Buffer :", EPISODE_REWARD_BUFFER)

if __name__ == '__main__':
    main()