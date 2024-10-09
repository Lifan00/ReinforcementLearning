import time
import torch

from hvac.entity.Home import Home
from hvac.tools.config import STATE_SPACE, ACTION_SPACE

class EnvCore:
    def __init__(self):
        self.state_space = STATE_SPACE
        self.action_space= ACTION_SPACE
        self.home = Home()
        self.state= self.home.get_state()
        self.done = False

    def reset(self):
        self.home.reset()
        self.done = False
        self.state = self.home.get_state()
        return self.get_state()

    def step(self, actions,upstairs_temperature,downstairs_temperature):
        try:
            if len(actions) == self.action_space:
                self.state, reward, done, info = self.home.do_simulate(actions,upstairs_temperature,downstairs_temperature)
                self.done = done
                return self.get_state(), reward, done, info
            else:
                raise ValueError("Action Space Error")
        except ValueError as e:
            print("Error Quit", repr(e))

    def get_state(self):
        return self.state[0:5] + self.state[7:13] + self.state[14:20] + self.state[21:27]

    def get_reward(self):
        return 0 if self.done else self.home.get_reward()

    def get_info(self):
        return [] if self.done else self.home.get_info()

    def get_temperature(self):
        return [] if self.done else self.home.get_temperature()

    @staticmethod
    def get_done():
        return False
