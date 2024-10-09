class People:
    def __init__(self, id,seed=8,work_time=12):
        """
        0 -> Elder    ALL DAY \n
        1 -> children 7-17 \n
        2 -> adult1   8-18 \n
        3 -> adult2   9-19 \n
        Args:
            id: 1 or 2 or 3
        """
        self.id = id
        self.seed=seed  #不同房间 不同的开始工作时长
        self.work_time=work_time #不同房间 不同的工作时长

    # def is_in_room(self, time, id=-1):
    #     """
    #     Args:
    #         time: hours
    #         id: you can get an unreal people and give id to charge is_in_room
    #     Returns: is the id of the person in the room
    #     """
    #     id = self.id if id == -1 else id
    #     base_time = 6 + id
    #     return 0 if base_time < time < base_time + 10 else 1

    def is_in_room(self, time):
        """
        Args:
            time: hours
            id: you can get an unreal people and give id to charge is_in_room
        Returns: is the id of the person in the room
        """
        return 0 if self.seed < time < self.seed + self.work_time else 1

    def get_effect(self, time):
        result = time * self.id
        return 0 * result
