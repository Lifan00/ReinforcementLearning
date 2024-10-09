import datetime
import pickle
from statistics import mean

import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from hvac.entity.EnvCore import EnvCore
from hvac.tools.config import RATIO_KWH2JOULE
from rl_algorithm.rl_one.RDQN_net import RDQN


def rdqn_train():
    EPISODES = 1
    env = EnvCore()
    save_list = []
    save_list_state = []
    state_action_dict = {"state": [], "action": [], "reward": [], "done": [], "next_state": []}
    action_dict = {"0": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   "1": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                   "2": [-1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1]}

    # 用于结果的展示
    all_epsd_mean_pmv_datalist = []  # 每个episode的pmv均值
    all_epsd_mean_reward_datalist = []  # 每个episode的reward均值
    all_epsd_final_power_datalist = []  # 每个episode的power均值
    all_epsd_month_power_datalist = []

    agent = RDQN(chkpt_dir="", best_chkpt_dir="")
    best_score = 0

    # 进行实验
    for i_episode in tqdm(range(EPISODES)):  # 进行EPISODES轮train
        state = env.reset()  # 环境初始化并返回state
        counter = 0  # 统计运行的步数
        done = False

        # 用于统计的参数
        i_episode_pmv_accumulation = 0
        i_episode_power_accumulation = 0
        i_episode_reward_accumulation = 0

        i_episode_month_power_accumulation = []

        while not done:
            if counter % 15 == 0:
                action = agent.network.select_action(torch.FloatTensor(state).unsqueeze(0))
                next_state, reward, done, info = env.step(action_dict[str(action)])
                agent.learn(state, action, reward, next_state, done)
                state_action_dict["state"].append(state)
                state_action_dict["action"].append(action)
                state_action_dict["reward"].append(reward)
                state_action_dict["done"].append(done)
                state_action_dict["next_state"].append(next_state)

            else:
                action = action
                next_state, reward, done, info = env.step(action_dict[str(action)])
            counter += 1
            if done:
                break
            state = next_state

            # calc common statistics
            i_episode_pmv_accumulation += mean([abs(x) for x in env.get_info()[0]['pmv']])
            i_episode_reward_accumulation += reward
            i_episode_power_accumulation += sum(env.get_info()[0]['acp'])
            ts = env.get_info()[0]['ts'][0]  # 时间戳

            dt = datetime.datetime.fromtimestamp(ts)
            if dt.month != len(i_episode_month_power_accumulation):  # 到达该月份的能量追加的年的列表中
                i_episode_month_power_accumulation.append(i_episode_power_accumulation / RATIO_KWH2JOULE)

            # show msg
            if counter % 100000 == 0:
                print("\n================================")
                print(f"\tNow Counter is {counter}")
                print(f"\tPmv info is {i_episode_pmv_accumulation / counter}")
                print(f"\tReward info is {i_episode_reward_accumulation / counter}")
                print(f"\tPower Accum is {i_episode_power_accumulation}")

        if i_episode_reward_accumulation > best_score:
            best_score = i_episode_reward_accumulation
            agent.save_best_models()

        if i_episode + 1 == EPISODES:
            agent.save_models()

        # save common statistics
        all_epsd_mean_pmv_datalist.append(i_episode_pmv_accumulation / counter)
        all_epsd_mean_reward_datalist.append(i_episode_reward_accumulation / counter)
        all_epsd_final_power_datalist.append(i_episode_power_accumulation)

        # 修订every year 的每个月的能耗
        i_episode_month_power_accumulation = [
            i_episode_month_power_accumulation[i + 1] - i_episode_month_power_accumulation[i] if i != 0 else
            i_episode_month_power_accumulation[i + 1] for i in range(len(i_episode_month_power_accumulation) - 1)]
        all_epsd_month_power_datalist.append(i_episode_month_power_accumulation)  # 插入到总表中

        print(f"Mean pmv: {all_epsd_mean_pmv_datalist[-1]}")
        print(f"Mean reward: {all_epsd_mean_reward_datalist[-1]}")
        print(f"Final Power: {all_epsd_final_power_datalist[-1] / RATIO_KWH2JOULE}")
        print(f"Month Mean Power:{all_epsd_month_power_datalist}")
    print(all_epsd_month_power_datalist)

    save_list.append({
        "all_epsd_mean_pmv_datalist:": all_epsd_mean_pmv_datalist,
        "all_epsd_mean_reward_datalist": all_epsd_mean_reward_datalist,
        "all_epsd_final_power_datalist": all_epsd_final_power_datalist,
        "all_epsd_month_power_datalist": all_epsd_month_power_datalist,
    })
    save_list_state.append(state_action_dict)
    save_data(save_list, "D:/FullMeta/scripts/data/theer_data.pkl")
    save_data(save_list_state, f"D:/FullMeta/scripts/data/{EPISODES}_data.pkl")


def rdqn_test():
    global months
    EPISODES = 3
    env = EnvCore()
    save_list = []
    all_epsd_data = []
    action_dict = {"0": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   "1": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                   "2": [-1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1]}

    # 用于结果的展示
    all_epsd_mean_pmv_datalist = []  # 每个episode的pmv均值
    all_epsd_mean_reward_datalist = []  # 每个episode的reward均值
    all_epsd_final_power_datalist = []  # 每个episode的power均值
    all_epsd_people_temp_datalist = []
    all_epsd_no_people_temp_datalist = []
    all_epsd_people_pmv_datalist = []
    all_epsd_no_people_pmv_datalist = []
    all_epsd_month_mean_no_people_temp_datalist = []
    all_epsd_month_mean_people_temp_datalist = []
    all_epsd_month_mean_temp_datalist = []
    all_epsd_month_mean_power_datalist = []

    agent = RDQN(chkpt_dir="D:/FullMeta/scripts/data/tmp/old", best_chkpt_dir="D:/FullMeta/scripts/data/tmp/best")

    agent.load_models()

    # 进行实验
    for _ in tqdm(range(EPISODES)):  # 进行EPISODES轮train
        state = env.reset()  # 环境初始化并返回state
        counter = 0  # 统计运行的步数
        done = False

        # 用于统计的参数
        i_episode_pmv_accumulation = 0
        i_episode_power_accumulation = 0
        i_episode_reward_accumulation = 0
        i_episode_people_temp = 0
        i_episode_no_people_temp = 0
        i_episode_people_pmv = 0
        i_episode_no_people_pmv = 0
        i_episode_month_mean_temp = []
        i_episode_month_mean_people_temp = []
        i_episode_month_mean_no_people_temp = []
        i_episode_month_mean_power = []

        i_episode_data = {"state": [], "action": [], "reward": [], "done": [], "next_state": []}

        # Define a list of room names in the desired order
        room_names = ["room0", "room1", "room2", "room3"]
        months = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"]
        # Initialize dictionaries to store temperature and pmv data for each room
        i_episode_room_data_pmv_tmp = {
            room_name: {"people_temp": [], "people_pmv": [], "no_people_temp": [], "no_people_pmv": [],
                        "month_temp": {month: {"temp": [], "people_temp": [], "no_people_temp": [], "power": []} for
                                       month in months},
                        } for room_name in room_names}

        while not done:
            action = agent.network.select_action(torch.FloatTensor(state).unsqueeze(0))
            next_state, reward, done, info = env.step(action_dict[str(action)])
            i_episode_data["state"].append(state)
            i_episode_data["action"].append(action)
            i_episode_data["reward"].append(reward)
            i_episode_data["done"].append(done)
            i_episode_data["next_state"].append(next_state)

            counter += 1
            if done:
                break
            state = next_state

            # calc common statistics
            i_episode_power_accumulation += sum(env.get_info()[0]['acp'])
            i_episode_pmv_accumulation += mean([abs(x) for x in env.get_info()[0]['pmv']])
            i_episode_reward_accumulation += reward

            ts = env.get_info()[0]['ts'][0]  # 时间戳
            dt = datetime.datetime.fromtimestamp(ts)

            # Iterate through the rooms and append temperature and pmv values
            hour_ranges = [(0, 0), (7, 17), (8, 18), (9, 19)]
            # Iterate through the rooms and append temperature and pmv values
            for i, hour_range in enumerate(hour_ranges):
                i_episode_room_data_pmv_tmp[room_names[i]]["month_temp"][f"{dt.month}"]["temp"].append(
                    env.get_info()[0]['temperature'][i])
                i_episode_room_data_pmv_tmp[room_names[i]]["month_temp"][f"{dt.month}"]["power"].append(
                    env.get_info()[0]['acp'][i])
                if hour_range[0] < dt.hour < hour_range[1]:
                    i_episode_room_data_pmv_tmp[room_names[i]]["no_people_temp"].append(
                        env.get_info()[0]['temperature'][i])
                    i_episode_room_data_pmv_tmp[room_names[i]]["no_people_pmv"].append(env.get_info()[0]['pmv'][i])
                    i_episode_room_data_pmv_tmp[room_names[i]]["month_temp"][f"{dt.month}"]["no_people_temp"].append(
                        env.get_info()[0]['temperature'][i])
                else:
                    i_episode_room_data_pmv_tmp[room_names[i]]["people_temp"].append(
                        env.get_info()[0]['temperature'][i])
                    i_episode_room_data_pmv_tmp[room_names[i]]["people_pmv"].append(env.get_info()[0]['pmv'][i])
                    i_episode_room_data_pmv_tmp[room_names[i]]["month_temp"][f"{dt.month}"]["people_temp"].append(
                        env.get_info()[0]['temperature'][i])

            # show msg
            if counter % 100000 == 0:
                print("\n================================")
                print(f"\tNow Counter is {counter}")
                print(f"\tPmv info is {i_episode_pmv_accumulation / counter}")
                print(f"\tReward info is {i_episode_reward_accumulation / counter}")
                print(f"\tPower Accum is {i_episode_power_accumulation}")

        # 根据baseline参数计算
        for i in range(4):
            i_episode_people_temp += mean(x for x in i_episode_room_data_pmv_tmp[room_names[i]]["people_temp"])
            i_episode_people_pmv += mean(x for x in i_episode_room_data_pmv_tmp[room_names[i]]["people_pmv"])
        for i in range(3):
            i_episode_no_people_temp += mean(
                x for x in i_episode_room_data_pmv_tmp[room_names[i + 1]]["no_people_temp"])
            i_episode_no_people_pmv += mean(x for x in i_episode_room_data_pmv_tmp[room_names[i + 1]]["no_people_pmv"])

        for month in months:
            i_episode_room_month_mean_temp = 0
            i_episode_room_month_mean_people_temp = 0
            i_episode_room_month_mean_no_people_temp = 0
            i_episode_room_month_mean_power = 0
            for i in range(4):
                i_episode_room_month_mean_temp += mean(
                    x for x in i_episode_room_data_pmv_tmp[room_names[i]]["month_temp"][month]["temp"])
                i_episode_room_month_mean_people_temp += mean(
                    x for x in i_episode_room_data_pmv_tmp[room_names[i]]["month_temp"][month]["people_temp"])
                i_episode_room_month_mean_power += mean(
                    x for x in i_episode_room_data_pmv_tmp[room_names[i]]["month_temp"][month]["power"])
            for i in range(3):
                i_episode_room_month_mean_no_people_temp += mean(
                    x for x in i_episode_room_data_pmv_tmp[room_names[i + 1]]["month_temp"][month]["no_people_temp"])
            i_episode_month_mean_temp.append(i_episode_room_month_mean_temp / 4)
            i_episode_month_mean_people_temp.append(i_episode_room_month_mean_people_temp / 4)
            i_episode_month_mean_no_people_temp.append(i_episode_room_month_mean_no_people_temp / 3)
            i_episode_month_mean_power.append(i_episode_room_month_mean_power / 4)

        all_epsd_data.append(i_episode_data)

        # save common statistics
        all_epsd_final_power_datalist.append(i_episode_power_accumulation / RATIO_KWH2JOULE)
        all_epsd_mean_pmv_datalist.append(i_episode_pmv_accumulation / counter)
        all_epsd_mean_reward_datalist.append(i_episode_reward_accumulation / counter)
        all_epsd_people_temp_datalist.append(i_episode_people_temp / 4)
        all_epsd_no_people_temp_datalist.append(i_episode_no_people_temp / 3)
        all_epsd_people_pmv_datalist.append(i_episode_people_pmv / 4)
        all_epsd_no_people_pmv_datalist.append(i_episode_no_people_pmv / 3)
        all_epsd_month_mean_temp_datalist.append(i_episode_month_mean_temp)
        all_epsd_month_mean_people_temp_datalist.append(i_episode_month_mean_people_temp)
        all_epsd_month_mean_no_people_temp_datalist.append(i_episode_month_mean_no_people_temp)
        all_epsd_month_mean_power_datalist.append(i_episode_month_mean_power)

        print(f"DATA:{all_epsd_data[-1]}")

        print(f"Final Power: {all_epsd_final_power_datalist[-1]}")
        print(f"Mean pmv: {all_epsd_mean_pmv_datalist[-1]}")
        print(f"Mean reward: {all_epsd_mean_reward_datalist[-1]}")
        print(f"Peole temp:{all_epsd_people_temp_datalist[-1]}")
        print(f"No Peole temp:{all_epsd_no_people_temp_datalist[-1]}")
        print(f"Peole pmv:{all_epsd_people_pmv_datalist[-1]}")
        print(f"No Peole pmv:{all_epsd_no_people_pmv_datalist[-1]}")
        print(f"Month Mean temp:{all_epsd_month_mean_temp_datalist[-1]}")
        print(f"Month People Mean Temperature:{all_epsd_month_mean_people_temp_datalist[-1]}")
        print(f"Month No People Mean Temperature:{all_epsd_month_mean_no_people_temp_datalist[-1]}")
        print(f"Month Mean Power:{all_epsd_month_mean_power_datalist[-1]}")

    save_list.append({
        "Final Power": all_epsd_final_power_datalist,
        "Mean pmv": all_epsd_mean_pmv_datalist,
        "Mean reward": all_epsd_mean_reward_datalist,
        "Peole temp": all_epsd_people_temp_datalist,
        "No Peole temp": all_epsd_people_pmv_datalist,
        "Peole pmv:": all_epsd_people_pmv_datalist,
        "No Peole pmv": all_epsd_no_people_pmv_datalist,
        "Month Mean temp": all_epsd_month_mean_temp_datalist,
        "Month People Mean Temperature": all_epsd_month_mean_people_temp_datalist,
        "Month No People Mean Temperature": all_epsd_month_mean_no_people_temp_datalist,
        "Month Mean Power": all_epsd_month_mean_power_datalist
    })

    save_data(save_list, filename=f"D:/FullMeta/scripts/data/{EPISODES}_epsd_baseline_data.pkl")
    save_data(all_epsd_data, filename=f"D:/FullMeta/scripts/data/{EPISODES}_[state_action_reward_done_next_state].pkl")

    plot_learning_bar(months, all_epsd_month_mean_power_datalist[-1], "D:/FullMeta/scripts/data/plots/month_power.png")


def plot_learning_curve(x, scores, figure_file):
    plt.plot(x, scores)
    plt.title('Running rainbow_dqn scores ---train')
    plt.savefig(figure_file)


def plot_learning_bar(x, scores, figure_file):
    # 创建柱状图
    plt.bar(x, scores)

    # 添加标题和标签
    plt.xlabel('Months')
    plt.ylabel('Power')
    plt.title('Monthly Average Power')
    plt.savefig(figure_file)
    # 显示柱状图
    plt.show()


def save_data(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)


if __name__ == '__main__':
    # rdqn_train()
    rdqn_test()

# 温度、最好最后、state
