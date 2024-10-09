import csv
import os
import pickle
from datetime import datetime

import numpy as np

from hvac.entity.People import People
from hvac.tools.config import RATIO_KWH2JOULE
from hvac.tools.tools import stamp_2_month, stamp_2_hour


class PklRecorder:
    def __init__(self, filename="",n_agents=1):
        self._on_recording = False
        self.data = []
        self.n_agents=n_agents
        #写个分支语句区分单双智能体
        self.traj = {i:{
            "observations": [],
            "actions": [],
            "rewards": [],
            "terminals": [],
            "next_observations": []
        }for i in range(self.n_agents)}

        self.data_transfer_to_array()
        self.init_timestamp = datetime.now().strftime("%y-%m-%d-%H-%M-%S")
        self.filename = filename

    def collect(self, state, action, reward, done, next_state):
        for i in range(self.n_agents):
            self.traj[i]["observations"].append(state[i])
            self.traj[i]["actions"].append(action[i])
            self.traj[i]["rewards"].append(reward[i])
            self.traj[i]["terminals"].append(done[i])
            self.traj[i]["next_observations"].append(next_state[i])

    def data_transfer_to_array(self):
        data_array = {i:{
            "actions": np.array(self.traj[i]["actions"], dtype=np.float32),
            "observations": np.array(self.traj[i]["observations"], dtype=np.float32),
            "next_observations": np.array(self.traj[i]["next_observations"], dtype=np.float32),
            "rewards": np.array(self.traj[i]["rewards"], dtype=np.float32),
            "terminals": np.array(self.traj[i]["terminals"])
        }for i in range(self.n_agents)}
        return data_array

    def start(self, flag=False):
        self._on_recording = flag

    def end(self):
        self._on_recording = False
        self.data.append(self.data_transfer_to_array())
        self.traj = {i:{
            "observations": [],
            "actions": [],
            "rewards": [],
            "terminals": [],
            "next_observations": []
        }for i in range(self.n_agents)}

    def save(self,dir_path = r"./outputs/"):
        os.makedirs(dir_path, exist_ok=True)
        path = os.path.join(dir_path, self.filename + "_" + self.init_timestamp + ".pkl")
        with open(path, 'wb') as file:
            pickle.dump(self.data, file)

class DataRecorder:
    """
    需要记录的数据：
    计算全年能耗
    计算全年平均PMV
    计算全年平均reward

    记录人在家和人不在家的温度
    记录人在家和人不在家的PMV

    记录每个月的平均温度
    记录每个月的能耗
    记录每个月人在家和人不在家的平均温度
    """

    def __init__(self, filename="", writeCSV=True):
        self._on_recording = False
        self.data = {
            'power': [[] for _ in range(12)],
            'reward': [[] for _ in range(12)],
            'temperature_in': [[] for _ in range(12)],
            'temperature_out': [[] for _ in range(12)],
            'pmv_in': [[] for _ in range(12)],
            'pmv_out': [[] for _ in range(12)]
        }
        self.init_timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        self.writeCSV = writeCSV
        if self.writeCSV:
            self.csvPath = self.createCSV(filename)

    def collect(self, info, reward):
        if self._on_recording:
            # 我这里的记录是每个房间都单独记录
            unrealPeople = People(-1)
            ts = info['ts'][0]
            month_index = stamp_2_month(ts) - 1
            hour = stamp_2_hour(ts)
            id_list = info['id']
            temperature_list = info['temperature']
            pmv_list = info['pmv']
            acp_list = info['acp']

            for id in map(int, id_list):
                # 记录能耗
                self.data['power'][month_index].append(acp_list[id])

                if unrealPeople.is_in_room(hour, id):
                    self.data['temperature_in'][month_index].append(temperature_list[id])
                    self.data['pmv_in'][month_index].append(abs(pmv_list[id]))
                else:
                    self.data['temperature_out'][month_index].append(temperature_list[id])
                    self.data['pmv_out'][month_index].append(abs(pmv_list[id]))
            # 记录reward
            self.data['reward'][month_index].append(reward)
        else:
            pass

    def print(self):
        # 全年能耗
        total_power = 0
        month_power = []
        for month_data in self.data['power']:
            month_power.append(round(sum(month_data) / RATIO_KWH2JOULE, 2))
            total_power += sum(month_data)

        # 全年reward
        yearly_average_reward = 0
        reward_len = 0
        for month_data in self.data['reward']:
            yearly_average_reward += sum(month_data)
            reward_len += len(month_data)
        yearly_average_reward = yearly_average_reward / reward_len if reward_len > 0 else 0
        summer_of_reward = yearly_average_reward * reward_len

        # 全年pmv
        all_pmv = []
        for month_index in range(12):
            all_pmv.extend(self.data['pmv_in'][month_index])
            all_pmv.extend(self.data['pmv_out'][month_index])
        yearly_average_pmv = sum(all_pmv) / len(all_pmv) if all_pmv else 0

        # 全年，人在和人不在的温度均值，人在和人不在的PMV均值
        yearly_average_temper_in = 0
        yearly_average_temper_out = 0
        yearly_average_pmv_in = 0
        yearly_average_pmv_out = 0
        len_of_temper_in = 0
        len_of_temper_out = 0
        len_of_pmv_in = 0
        len_of_pmv_out = 0
        for month_index in range(12):
            yearly_average_temper_in += sum(self.data['temperature_in'][month_index])
            yearly_average_temper_out += sum(self.data['temperature_out'][month_index])
            yearly_average_pmv_in += sum(self.data['pmv_in'][month_index])
            yearly_average_pmv_out += sum(self.data['pmv_out'][month_index])

            len_of_temper_in += len(self.data['temperature_in'][month_index])
            len_of_temper_out += len(self.data['temperature_out'][month_index])
            len_of_pmv_in += len(self.data['pmv_in'][month_index])
            len_of_pmv_out += len(self.data['pmv_out'][month_index])

        yearly_average_temper_in = yearly_average_temper_in / len_of_temper_in if len_of_temper_in > 0 else 0
        yearly_average_temper_out = yearly_average_temper_out / len_of_temper_out if len_of_temper_out > 0 else 0
        yearly_average_pmv_in = yearly_average_pmv_in / len_of_pmv_in if len_of_pmv_in > 0 else 0
        yearly_average_pmv_out = yearly_average_pmv_out / len_of_pmv_out if len_of_pmv_out > 0 else 0

        # 月均，人在和人不在的温度以及pmv
        get_mean = lambda sublist: round(sum(sublist) / len(sublist), 2) if sublist else 0
        monthly_average_temper_in = list(map(get_mean, self.data['temperature_in']))
        monthly_average_temper_out = list(map(get_mean, self.data['temperature_out']))
        monthly_average_pmv_in = list(map(get_mean, self.data['pmv_in']))
        monthly_average_pmv_out = list(map(get_mean, self.data['pmv_out']))

        print(f"全年能耗:{total_power / RATIO_KWH2JOULE:.2f}")
        print(f"全年总reward:{summer_of_reward:.2f}")
        print(f"全年平均reward:{yearly_average_reward:.2f}")
        print(f"全年平均pmv:{yearly_average_pmv:.2f}")
        print(f"人在平均温度:{yearly_average_temper_in:.2f}")
        print(f"人不在平均温度:{yearly_average_temper_out:.2f}")
        print(f"人在平均pmv:{yearly_average_pmv_in:.2f}")
        print(f"人不在平均pmv:{yearly_average_pmv_out:.2f}")
        print(f"月均人在平均温度:{monthly_average_temper_in}")
        print(f"月均人不在平均温度:{monthly_average_temper_out}")
        print(f"月均人在平均pmv:{monthly_average_pmv_in}")
        print(f"月均人不在平均pmv:{monthly_average_pmv_out}")
        print(f"月均能耗:{month_power}")

        # 写入csv
        if self.writeCSV:
            with open(self.csvPath, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                row = [
                    '{:.2f}'.format(total_power / RATIO_KWH2JOULE),
                    '{:.2f}'.format(summer_of_reward),
                    '{:.2f}'.format(yearly_average_reward),
                    '{:.2f}'.format(yearly_average_pmv),
                    '{:.2f}'.format(yearly_average_temper_in),
                    '{:.2f}'.format(yearly_average_temper_out),
                    '{:.2f}'.format(yearly_average_pmv_in),
                    '{:.2f}'.format(yearly_average_pmv_out),
                    monthly_average_temper_in,
                    monthly_average_temper_out,
                    monthly_average_pmv_in,
                    monthly_average_pmv_out,
                    month_power
                ]
                writer.writerow(row)
        self._recovery()
        # 测试功能
        return [yearly_average_pmv_in]

    def start(self, flag=False):
        self._on_recording = flag

    def end(self):
        self._on_recording = False

    def _recovery(self):
        self._on_recording = False
        self.data = {
            'power': [[] for _ in range(12)],
            'reward': [[] for _ in range(12)],
            'temperature_in': [[] for _ in range(12)],
            'temperature_out': [[] for _ in range(12)],
            'pmv_in': [[] for _ in range(12)],
            'pmv_out': [[] for _ in range(12)]
        }

    def createCSV(self, filename):
        dir_path = r"./outputs/"
        os.makedirs(dir_path, exist_ok=True)
        path = os.path.join(dir_path, filename + "_" + self.init_timestamp + ".csv")
        with open(path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            header = [
                '全年 能耗',
                '全年总 Reward',
                '平均 Reward',
                '平均 PMV',
                '平均 温度 (人在家)',
                '平均 温度 (人不在家)',
                '平均 PMV (人在家)',
                '平均 PMV (人不在家)',
                '月均 温度 (人在家)',
                '月均 温度 (人不在家)',
                '月均 PMV (人在家)',
                '月均 PMV (人不在家)',
                '月均 能耗 '
            ]
            writer.writerow(header)
        return path


class PmvBasedController:
    def __init__(self):
        self.TEMPERATURE_BEST_PMV0 = [
            21.9,  # 这个是12月，对应12%12=0的计算
            21.4,
            22.3,
            24.2,
            25.6,
            26.6,
            27.0,
            27.1,
            27.1,
            26.4,
            26.0,
            24.6,
            21.9
        ]
        self.differ_limit = 2
        self.unreal_people = People(-1)

    def select_action(self, state, ts):
        """
        PmvBasedController并不知道人在不在家
        """
        action = []
        room_id_list = [1, 2, 3, 0]
        temper_out = state[0]
        temper_in_list = [state[1], state[5], state[11], state[17]]
        month = stamp_2_month(ts)
        hour = stamp_2_hour(ts)
        for id in room_id_list:
            # 将 False 改为 True 可以禁用对于人员检测的控制，从而控制能耗
            people_exist = self.unreal_people.is_in_room(hour, id) or False
            temper_in = temper_in_list[id]
            best_temperature = self.TEMPERATURE_BEST_PMV0[(1 + month) % 12]
            # diverse > 0 means that now is too hot and needs cold
            diverse = temper_in - best_temperature
            if diverse == 0:
                # 当前状态很好，保温
                room_action = [0, 0, 0]
            elif diverse > 0:
                # 人需要降温
                if temper_in >= temper_out:
                    # 外面更凉快，直接通风
                    room_action = [0, 1, 1]
                else:
                    # 外面比屋内热
                    if abs(diverse) > self.differ_limit:
                        # 热超出了我们的忍受值，直接开制冷
                        room_action = [-1, 0, 1]
                    else:
                        # 在忍受范围内
                        if abs(diverse) >= abs(temper_in - temper_out):
                            # 屋内屋外都差不多，直接通风
                            room_action = [0, 1, 1]
                        else:
                            # 室内可以忍，但是室外太热不能忍，关窗保温
                            # todo: 考虑以后在这个档位开弱冷
                            room_action = [0, 0, 1]
            else:
                # 人需要保暖
                if temper_in <= temper_out:
                    # 外面更暖和，直接通风
                    room_action = [0, 1, 1]
                else:
                    # 外面比屋内更冷
                    if abs(diverse) > self.differ_limit:
                        # 太冷了，直接开制暖
                        room_action = [1, 0, 1]
                    else:
                        # 在忍受范围内
                        if abs(diverse) >= abs(temper_in - temper_out):
                            # 屋内屋外都差不多，直接通风省电
                            room_action = [0, 1, 1]
                        else:
                            # 室内可以忍，但是室外太冷，关窗保温
                            # todo: 考虑以后在这个档位开弱热
                            room_action = [0, 0, 1]
            room_action[0] = room_action[0] if people_exist else 0
            action += [room_action[0], room_action[2]] if id == 0 else room_action
        return action


if __name__ == "__main__":
    # action = []
    # room_list = [1, 2, 3, 0]
    # for i in room_list:
    #     room_action = [i * 4 + 1, i * 4 + 2, i * 4 + 3]
    #     action += [room_action[0], room_action[2]] if i == 0 else room_action
    #     # if i == 0:
    #     #     action += [room_action[0], room_action[2]]
    #     # else:
    #     #     action += room_action
    # print(action)

    get_mean = lambda sublist: sum(sublist) / len(sublist) if sublist else 0
    data = [
        [1, 2, 3],
        [4, 5, 6],
        [9, 8, 7]
    ]
    print(list(map(get_mean, data)))
