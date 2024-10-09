import random

import numpy as np

from hvac.entity.LivingRoom import LivingRoom
from hvac.entity.Room import Room
from hvac.tools.config import INIT_TIMESTAMP, WALL_THK, WALL_LAMBDA, LR_W, LR_L, ROOM_H, ROOM_L, ROOM_W, TIME_GAP
from hvac.tools.tools import import_temper, data_split, import_illu, calc_heat_energy


class Home:
    def __init__(self):
        self.time_stamp = INIT_TIMESTAMP
        self.room_list = [LivingRoom(0)]
        for id in range(1, 4):
            self.room_list.append(Room(id))

        self._data_temperature_outside = data_split(import_temper())
        self._data_illusion_outside = data_split(import_illu())
        self.data_size = len(self._data_temperature_outside)
        self.update_index = 0

    def reset(self):
        self.time_stamp = INIT_TIMESTAMP
        for room in self.room_list:
            room.reset()
        self.update_index = 0

    def do_simulate(self, actions,upstairs_temperature,downstairs_temperature):
        if self.update_index < self.data_size:
            _curr_state_const = self.get_state()
            _curr_state_const[0] = self._data_temperature_outside[self.update_index]

            # Room temperature index:
            # outside   LR  Room1   Room2    Room3
            # 0         1     7       14        21
            # every temperature difference is calculated by inside - outside
            # =================================================================
            # 0 Living Room
            lr_heat_energy = 0
            lr_td_list = [
                _curr_state_const[1] - _curr_state_const[0],  # LR and outside
                _curr_state_const[1] - _curr_state_const[7],  # LR and room1
                _curr_state_const[1] - _curr_state_const[14],  # LR and room2
                _curr_state_const[1] - _curr_state_const[21],  # LR and room3
                _curr_state_const[1] - upstairs_temperature[0], # LR and upstairs
                _curr_state_const[1] - downstairs_temperature[0] # LR and downstairs
            ]  # 内部温度减去外部温度，若td是正值，则房间整体能量流失，因此使用-。
            # # 0 Doors
            # lr_heat_energy -= self.room_list[0].door.get_conduct_heat(lr_td_list[0])
            # lr_heat_energy -= self.room_list[1].door.get_conduct_heat(lr_td_list[1])
            # lr_heat_energy -= self.room_list[2].door.get_conduct_heat(lr_td_list[2])
            # lr_heat_energy -= self.room_list[3].door.get_conduct_heat(lr_td_list[3])
            # 0 Walls
            lr_S_walls = [
                (LR_W * 2 + LR_L) * ROOM_H - self.room_list[0].door.area,
                ROOM_L * ROOM_H - self.room_list[1].door.area,
                ROOM_L * ROOM_H - self.room_list[2].door.area,
                ROOM_L * ROOM_H - self.room_list[3].door.area,
                ROOM_L * ROOM_H,
                ROOM_L * ROOM_H,
            ]
            lr_heat_energy -= calc_heat_energy(WALL_LAMBDA, lr_S_walls[0], lr_td_list[0], WALL_THK)
            lr_heat_energy -= calc_heat_energy(WALL_LAMBDA, lr_S_walls[1], lr_td_list[1], WALL_THK)
            lr_heat_energy -= calc_heat_energy(WALL_LAMBDA, lr_S_walls[2], lr_td_list[2], WALL_THK)
            lr_heat_energy -= calc_heat_energy(WALL_LAMBDA, lr_S_walls[3], lr_td_list[3], WALL_THK)
            lr_heat_energy -= calc_heat_energy(WALL_LAMBDA, lr_S_walls[4], lr_td_list[4], WALL_THK)
            lr_heat_energy -= calc_heat_energy(WALL_LAMBDA, lr_S_walls[5], lr_td_list[5], WALL_THK)
            self.room_list[0].update(actions[-2:], lr_heat_energy, lr_td_list)
            # =================================================================
            # 1 Room 1
            r1_heat_energy = 0
            r1_td_list = [
                _curr_state_const[7] - _curr_state_const[0],  # r1 and outside
                _curr_state_const[7] - _curr_state_const[1],  # r1 and lr
                _curr_state_const[7] - _curr_state_const[14],  # r1 and r2
                _curr_state_const[7] - upstairs_temperature[1],  # r1 and upstairs
                _curr_state_const[7] - downstairs_temperature[1]  # r1 and downstairs
            ]
            # # 1 Door & Window
            # r1_heat_energy -= self.room_list[1].door.get_conduct_heat(r1_td_list[1])
            # r1_heat_energy -= self.room_list[1].wd.get_conduct_heat(r1_td_list[0])
            # 1 Walls
            r1_S_walls = [
                ROOM_L * ROOM_H - self.room_list[1].door.area,  # r1 and lr
                ROOM_L * ROOM_H - self.room_list[1].wd.area + ROOM_W * ROOM_H,  # r1 and outside
                ROOM_W * ROOM_H,  # r1 and r2
                ROOM_L * ROOM_W,
                ROOM_L * ROOM_W,
            ]
            r1_heat_energy -= calc_heat_energy(WALL_LAMBDA, r1_S_walls[0], r1_td_list[1], WALL_THK)
            r1_heat_energy -= calc_heat_energy(WALL_LAMBDA, r1_S_walls[1], r1_td_list[0], WALL_THK)
            r1_heat_energy -= calc_heat_energy(WALL_LAMBDA, r1_S_walls[2], r1_td_list[2], WALL_THK)
            r1_heat_energy -= calc_heat_energy(WALL_LAMBDA, r1_S_walls[3], r1_td_list[3], WALL_THK)
            r1_heat_energy -= calc_heat_energy(WALL_LAMBDA, r1_S_walls[4], r1_td_list[4], WALL_THK)
            self.room_list[1].update(actions[0:3], r1_heat_energy, [r1_td_list[1], r1_td_list[0]])
            # =================================================================
            # 2 Room 2
            r2_heat_energy = 0
            r2_td_list = [
                _curr_state_const[14] - _curr_state_const[0],  # r2 and outside
                _curr_state_const[14] - _curr_state_const[1],  # r2 and lr
                _curr_state_const[14] - _curr_state_const[7],  # r2 and r1
                _curr_state_const[14] - _curr_state_const[21],  # r2 and r3
                _curr_state_const[14] - upstairs_temperature[2],  # r2 and upstairs
                _curr_state_const[14] - downstairs_temperature[2]  # r2 and downstairs
            ]
            # # 2 Door & Window
            # r2_heat_energy -= self.room_list[2].door.get_conduct_heat(r2_td_list[1])
            # r2_heat_energy -= self.room_list[2].wd.get_conduct_heat(r2_td_list[0])
            # 2 Walls
            r2_S_walls = [
                ROOM_L * ROOM_H - self.room_list[2].door.area,
                ROOM_L * ROOM_H - self.room_list[2].wd.area,
                ROOM_W * ROOM_H,
                ROOM_W * ROOM_H,
                ROOM_L * ROOM_W,
                ROOM_L * ROOM_W,
            ]
            r2_heat_energy -= calc_heat_energy(WALL_LAMBDA, r2_S_walls[0], r2_td_list[1], WALL_THK)
            r2_heat_energy -= calc_heat_energy(WALL_LAMBDA, r2_S_walls[1], r2_td_list[0], WALL_THK)
            r2_heat_energy -= calc_heat_energy(WALL_LAMBDA, r2_S_walls[2], r2_td_list[2], WALL_THK)
            r2_heat_energy -= calc_heat_energy(WALL_LAMBDA, r2_S_walls[3], r2_td_list[3], WALL_THK)
            r2_heat_energy -= calc_heat_energy(WALL_LAMBDA, r2_S_walls[4], r2_td_list[4], WALL_THK)
            r2_heat_energy -= calc_heat_energy(WALL_LAMBDA, r2_S_walls[5], r2_td_list[5], WALL_THK)
            self.room_list[2].update(actions[3:6], r2_heat_energy, [r2_td_list[1], r2_td_list[0]])
            # =================================================================
            # 3 Room 3
            r3_heat_energy = 0
            r3_td_list = [
                _curr_state_const[21] - _curr_state_const[0],  # r3 and outside
                _curr_state_const[21] - _curr_state_const[1],  # r3 and lr
                _curr_state_const[21] - _curr_state_const[14],  # r3 and r2
                _curr_state_const[21] - upstairs_temperature[3],  # r3 and upstairs
                _curr_state_const[21] - downstairs_temperature[3]  # r3 and downstairs
            ]
            # # 3 Door & Window
            # r3_heat_energy -= self.room_list[3].door.get_conduct_heat(r3_td_list[1])
            # r3_heat_energy -= self.room_list[3].wd.get_conduct_heat(r3_td_list[0])
            # 3 Walls
            r3_S_walls = [
                ROOM_L * ROOM_H - self.room_list[3].door.area,
                ROOM_L * ROOM_H - self.room_list[3].wd.area + ROOM_W * ROOM_H,
                ROOM_W * ROOM_H,
                ROOM_W * ROOM_L,
                ROOM_W * ROOM_L,
            ]
            r3_heat_energy -= calc_heat_energy(WALL_LAMBDA, r3_S_walls[0], r3_td_list[1], WALL_THK)
            r3_heat_energy -= calc_heat_energy(WALL_LAMBDA, r3_S_walls[1], r3_td_list[0], WALL_THK)
            r3_heat_energy -= calc_heat_energy(WALL_LAMBDA, r3_S_walls[2], r3_td_list[2], WALL_THK)
            r2_heat_energy -= calc_heat_energy(WALL_LAMBDA, r2_S_walls[3], r2_td_list[3], WALL_THK)
            r2_heat_energy -= calc_heat_energy(WALL_LAMBDA, r2_S_walls[4], r2_td_list[4], WALL_THK)
            self.room_list[3].update(actions[6:9], r3_heat_energy, [r3_td_list[1], r3_td_list[0]])
        # =================================================================
        self.time_stamp += TIME_GAP
        self.update_index += 1
        if self.update_index < self.data_size:
            state = self.get_state()
            info = self.get_info()
            # info="You should Reset the Environment"
            reward = self.get_reward()
            done = self.get_done()
        else:
            return [], -1, True, "You should Reset the Environment"
        # =================================================================

        return state, reward, done, info

    def get_state(self):
        # A & B
        state = [self._data_temperature_outside[self.update_index]]
        for room in self.room_list:
            state = state + room.get_state()
        return state

    def get_reward(self):
        reward_list = [room.get_reward() for room in self.room_list]
        return sum(reward_list) / len(reward_list)

    def get_done(self):
        done = False if self.update_index < self.data_size else True
        return done

    def get_temperature(self):
        temperature = np.array([tepr.temperature_inside for tepr in self.room_list])
        return temperature

    def get_info(self):
        info_arr = np.array([tepr.get_info()[0] for tepr in self.room_list])
        info = {
            'id': info_arr[:, 0],
            'temperature': info_arr[:, 1],
            'pmv': info_arr[:, 2],
            'ac': info_arr[:, 3],
            'door': info_arr[:, 4],
            'wd': info_arr[:, 5],
            'acp': info_arr[:, 6],
            'ts': info_arr[:, 7]
        }
        msg = "\n" + "Temperature Outside： " + str(self._data_temperature_outside[self.update_index]) + "".join(
            map(str, [tepr.get_info()[1] for tepr in self.room_list])
        )
        return info, msg