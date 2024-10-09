from collections import deque

from hvac.entity.AirConditioner import AirConditioner
from hvac.entity.Door import Door
from hvac.entity.Room import Room
from hvac.tools.config import ROOM_H, INIT_TEMPERATURE_IN, DEFAULT_VEL, \
     DEFAULT_MET, DEFAULT_PA, INIT_TIMESTAMP, AIR_DENSITY, AIR_SHC, LR_L, LR_W, TIME_GAP, \
    PMV_LIMIT, DEQUE_SIZE
from hvac.tools.tools import computePMV, CLO_L, stamp_2_month, air_condition_restrictor, calculate_reward, stamp_2_hour


class LivingRoom(Room):
    def __init__(self, id):
        super().__init__(id)
        self.id = id
        self.L = LR_L
        self.W = LR_W
        self.H = ROOM_H
        self.M = LR_L * LR_W * ROOM_H * AIR_DENSITY
        self.time_stamp = INIT_TIMESTAMP
        # people在reward中直接使用people_exist进行计算
        # 默认客厅是老人，每天任何时候都在家
        # self.people = People(id)

        self.ac = AirConditioner()
        self.door = Door()

        self.temperature_inside = INIT_TEMPERATURE_IN

        self.power_consumption = 0
        self.pmv = 0
        self.power_queue = deque(maxlen=DEQUE_SIZE)

    def reset(self):
        self.L = LR_L
        self.W = LR_W
        self.H = ROOM_H
        self.M = LR_L * LR_W * ROOM_H * AIR_DENSITY
        self.time_stamp = INIT_TIMESTAMP

        self.ac.reset()
        self.door.reset()

        self.temperature_inside = INIT_TEMPERATURE_IN

        self.power_consumption = 0
        self.pmv = 0
        self.power_queue.clear()

    def update(self, actions, heat_energy_d, temperature_list,DEFAULT_RH):
        """
        Args:
            actions: [ac,door]
            heat_energy_d: as its name
            temperature_list: 4 door temperature list [0,1,2,3]
        """

        # 1.change action

        self.door.set_state(actions[1])

        # 2.update 4 door
        heat_energy_d -= self.door.get_conduct_heat(temperature_list[0])
        heat_energy_d -= self.door.get_conduct_heat(temperature_list[1])
        heat_energy_d -= self.door.get_conduct_heat(temperature_list[2])
        heat_energy_d -= self.door.get_conduct_heat(temperature_list[3])

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
        # return [self.temperature_inside]

        # B
        return [self.temperature_inside, self.ac.get_state(), self.door.get_state(),
                LR_L * LR_W * ROOM_H, 0, 0]
        # pass

    def get_reward(self):
        people_exist = True
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
            -999,
            self.ac.get_power() * TIME_GAP,
            self.time_stamp
        ]
        msg = "\n " + "=====LivingRoom {} =====".format(self.id)
        msg += "\n " + "Temperature_inside: {}".format(self.temperature_inside)
        msg += "\n " + "PMV: {}".format(self.pmv)
        msg += "\n " + "AirCondition State: {}".format(self.ac.get_state())
        msg += "\n " + "Door State: {}".format(self.door.get_state())
        msg += "\n " + "AirCondition Power: {}".format(self.ac.get_power() * TIME_GAP)
        msg += "\n " + "Time Stamp: {}".format(self.time_stamp)
        return info, msg
