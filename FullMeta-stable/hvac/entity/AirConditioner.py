from hvac.tools.config import AC_LAMBDA, AC_POWER, AC_GENERATE


class AirConditioner:

    def __init__(self):
        self.gamma = AC_LAMBDA
        self.power_state = 0
        # this state is 1-5 inside

    def set_state(self, action):
        """
        Args:
            action: set power state to -2 to 2
        """
        self.power_state = action + 2

    def get_state(self):
        """
        Returns:
            power state from -2 to 2
        """
        return self.power_state - 2

    def get_power(self):
        return AC_POWER[int(self.power_state)]

    def get_heat_energy(self):
        return AC_GENERATE[int(self.power_state)]

    def reset(self):
        self.power_state = 0


if __name__ == "__main__":
    ac = AirConditioner()
    ac.set_state(-1)
    print(ac.get_state())
    print(ac.get_power())
    print(ac.get_heat_energy())
