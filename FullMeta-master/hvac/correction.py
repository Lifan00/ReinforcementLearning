import os
from time import sleep

from tqdm import trange

from hvac.entity.EnvCore import EnvCore
from hvac.tools.scripts import PmvBasedController

a1_action_close = [
    0, 0, 0,
    0, 0, 0,
    0, 0, 0,
    0, 0
]
a2_action_open_door = [
    0, 0, 1,
    0, 0, 1,
    0, 0, 1,
    0, 0
]
a3_action_open_wd = [
    0, 1, 0,
    0, 1, 0,
    0, 1, 0,
    0, 1
]
a4_action_open_door_and_wd = [
    0, 1, 1,
    0, 1, 1,
    0, 1, 1,
    0, 1
]
a5_action_open_ac = [
    1, 0, 0,
    1, 0, 0,
    1, 0, 0,
    1, 0
]

action_dist = [
    a1_action_close,
    a2_action_open_door,
    a3_action_open_wd,
    a4_action_open_door_and_wd,
    a5_action_open_ac
]

if __name__ == '__main__':
    os.chdir("../")
    env = EnvCore()
    EPISODES = 2
    pbc = PmvBasedController()

    with trange(EPISODES) as tqdm_episodes:
        for i_episode in tqdm_episodes:

            state = env.reset()
            counter = 0
            done = False

            while not done:
                print("---------------参数测试信息-----------------")
                auto = True
                if auto:
                    action = pbc.select_action(state, env.get_info()[0]['ts'][0])
                    print(action)
                else:
                    try:
                        choose = int(input("\n Choose the action"
                                           "\n 1: All closed"
                                           "\n 2: Open door"
                                           "\n 3: Open window"
                                           "\n 4: Open door and window"
                                           "\n 5: Open ac"
                                           "\n"))
                        action = action_dist[choose - 1]
                    except ValueError:
                        choose = 1
                        action = action_dist[choose - 1]

                state, reward, done, info = env.step(action)
                if done:
                    print("Step done")
                    sleep(5)
                env_info = env.get_info()[0]
                print(f"Temper Out:{state[0]:.2f},\tTemper Arr:{env_info['temperature']}")
                # print("\ntemperature arr: ", env_info['temperature'])
                counter += 1
                print(f"Done:{done}")
                print(f"Counter:{counter}")
                # print("\t counter {}".format(counter))
                # print("\n")
                # print("---------------内部测试信息-----------------")
                # print("\nreward: ", env.get_reward())
                # print("\ninfo: ", env.get_info()[1])
                # print("\nstate: ", env.get_state())
                # print("\ninfo arr: ", env.get_info()[0])
                # print("\n\t counter {}".format(counter))
                # print("\n\n")
