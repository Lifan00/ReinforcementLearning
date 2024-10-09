from collections import deque

from hvac.entity.AirConditioner import AirConditioner
from hvac.entity.Door import Door
from hvac.entity.People import People
from hvac.entity.Window import Window
from hvac.tools.config import ROOM_L, ROOM_W, ROOM_H, INIT_TEMPERATURE_IN, DEFAULT_VEL, \
    DEFAULT_RH, DEFAULT_MET, DEFAULT_PA, INIT_TIMESTAMP, AIR_DENSITY, AIR_SHC, TIME_GAP, \
    PMV_LIMIT, DEQUE_SIZE
from hvac.tools.tools import computePMV, CLO_L, stamp_2_month, stamp_2_hour, air_condition_restrictor, calculate_reward

class Room:
    def __init__(self, id):
        self.id = id
        self.L = ROOM_L
        self.W = ROOM_W
        self.H = ROOM_H
        self.M = ROOM_L * ROOM_W * ROOM_H * AIR_DENSITY
        self.time_stamp = INIT_TIMESTAMP
        self.people = People(id)

        self.ac = AirConditioner()
        self.wd = Window()
        self.door = Door()

        self.temperature_inside = INIT_TEMPERATURE_IN

        self.power_consumption = 0
        self.pmv = 0
        self.power_queue = deque(maxlen=DEQUE_SIZE)

    def reset(self):
        self.L = ROOM_L
        self.W = ROOM_W
        self.H = ROOM_H
        self.M = ROOM_L * ROOM_W * ROOM_H * AIR_DENSITY
        self.time_stamp = INIT_TIMESTAMP

        self.ac.reset()
        self.wd.reset()
        self.door.reset()

        self.temperature_inside = INIT_TEMPERATURE_IN

        self.power_consumption = 0
        self.pmv = 0
        self.power_queue.clear()

    def update(self, actions, heat_energy_d, temperature_list):
        """
        Args:
            actions: [ac,wd,door]
            heat_energy_d: as its name
            temperature_list: door and window temperature [door,window]
        """

        # 1.change action
        self.wd.set_state(actions[1])
        self.door.set_state(actions[2])

        # 2.update door and window
        heat_energy_d -= self.door.get_conduct_heat(temperature_list[0])
        heat_energy_d -= self.wd.get_conduct_heat(temperature_list[1])

        # 3.update air condition
        air_condition_action = air_condition_restrictor(actions[0], self.temperature_inside)
        self.ac.set_state(air_condition_action)
        heat_energy_d += self.ac.get_heat_energy() * TIME_GAP  # 空调是给房间提供热量，所以是+

        # 4.update temperature and time
        dt = heat_energy_d / (self.M * AIR_SHC)
        self.temperature_inside += dt
        self.time_stamp += TIME_GAP

        # 5.calculate and save statistics
        TA = self.temperature_inside
        month = stamp_2_month(self.time_stamp)
        people_effect = self.people.get_effect(stamp_2_hour(self.time_stamp))
        # todo VEL和风速相关，需要加入
        pmv = computePMV(TA, DEFAULT_VEL, DEFAULT_RH, DEFAULT_MET + people_effect, CLO_L[month], DEFAULT_PA)
        self.pmv = max(-PMV_LIMIT, min(PMV_LIMIT, pmv))

        # 6.Maintain queues
        while len(self.power_queue) < DEQUE_SIZE:
            self.power_queue.append(self.ac.get_power())
        self.power_queue.append(self.ac.get_power())

        # 7.save power
        self.power_consumption += (self.ac.get_power() * TIME_GAP)

        return self.temperature_inside, dt

    def get_state(self):
        # A
        # return [self.temperature_inside, self.people.is_in_room(stamp_2_hour(self.time_stamp))]

        # B
        return [self.temperature_inside, self.people.is_in_room(stamp_2_hour(self.time_stamp)), self.ac.get_state(),
                self.wd.get_state(), self.door.get_state(), ROOM_L * ROOM_W * ROOM_H, 0]
        # pass

    def get_reward(self):
        people_exist = self.people.is_in_room(stamp_2_hour(self.time_stamp))

        reward, _ = calculate_reward(list(self.power_queue), self.temperature_inside, abs(self.pmv), people_exist,
                                     self.time_stamp)
        return reward

    def _get_power_consumption(self):
        return self.power_consumption

    def _get_pmv(self):
        return self.pmv

    def get_info(self):
        info = [
            self.id,
            self.temperature_inside,
            self.pmv,
            self.ac.get_state(),
            self.door.get_state(),
            self.wd.get_state(),
            self.ac.get_power() * TIME_GAP,
            self.time_stamp
        ]
        msg = "\n " + "=====Room {} =====".format(self.id)
        msg += "\n " + "Temperature_inside: {}".format(self.temperature_inside)
        msg += "\n " + "PMV: {}".format(self.pmv)
        msg += "\n " + "AirCondition State: {}".format(self.ac.get_state())
        msg += "\n " + "Door State: {}".format(self.door.get_state())
        msg += "\n " + "Window State: {}".format(self.wd.get_state())
        msg += "\n " + "AirCondition Power: {}".format(self.ac.get_power() * TIME_GAP)
        msg += "\n " + "Time Stamp: {}".format(self.time_stamp)
        return info, msg
