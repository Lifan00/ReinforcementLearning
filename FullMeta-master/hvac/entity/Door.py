from hvac.tools.config import DOOR_L, DOOR_W, DOOR_LAMBDA, DOOR_THK, AIR_LAMBDA
from hvac.tools.tools import calc_heat_energy


class Door:
    def __init__(self):
        self.area = DOOR_L * DOOR_W
        self.gamma = DOOR_LAMBDA
        self.thk = DOOR_THK
        self.state = 0

    def get_conduct_heat(self, temperature_d):
        gamma = self.gamma + self.state * (AIR_LAMBDA * self.thk - self.gamma)
        return calc_heat_energy(gamma, self.area, temperature_d, self.thk)

    def reset(self):
        self.state = 0

    def set_state(self, action):
        self.state = action

    def get_state(self):
        return self.state
