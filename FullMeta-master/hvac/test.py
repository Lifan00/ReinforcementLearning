import time

import numpy as np


# while True:
#     TA = int(input("Enter"))
#     print(computePMV(TA, DEFAULT_VEL, DEFAULT_RH, DEFAULT_MET, CLO_L[1], DEFAULT_PA))


# 定义一个转换函数，入参为当前时间time.time()
def time_s_date(ts):
    dt = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(float(ts))))
    return dt


class RL:
    def __init__(self, id):
        self.id = id

    def get_info(self):
        return "\n " + str(self.id)


def rational(a, b, c, d):
    tmp_list = [a, b, c, d]
    lmax = max(tmp_list)
    lmin = min(tmp_list)
    return lmax - lmin != 2


buffer = []


def convert_json_key(param_dict):
    """
    json.dump不支持key是int的dict，在编码存储的时候会把所有的int型key写成str类型的
    所以在读取json文件后，用本方法将所有的被解码成str的int型key还原成int
    """
    new_dict = dict()
    for key, value in param_dict.items():
        try:
            new_key = int(key)
            new_dict[new_key] = value
        except:
            new_dict[key] = value

    return new_dict


def fuc_return_test():
    info = [0, time.time(), 1]
    msg = "test"
    return info, msg


if __name__ == '__main__':
    # env = EnvCore()
    # env.reset()
    # print(env.state_space)
    # action = [
    #     1, 1, 1,
    #     1, 1, 1,
    #     1, 1, 1,
    #     1, 1
    # ]
    # env.step(action)
    # state = env.get_state()
    # print("ok")
    # room_list = [RL(0), RL(4), RL(2), RL(3)]
    # msg = "\n str(self._data_temperature_outside)" + "".join(map(str, [tepr.get_info() for tepr in room_list]))
    # print(msg)
    # list1 = [1, 2, 3, 4]
    # print(list1)
    # for window0 in range(2):
    #     print(window0)
    # pass
    # while True:
    #     print(np.random.randint(2))
    # json_file = open("data/that_dict.json", "r")
    # data = json.load(json_file)
    # data = convert_json_key(data)
    # print(data)
    # print(data[0])
    # json_file.close()

    # print_file = open("test.txt", 'a+')
    # print("test 1\t" + str(time.time()), file=print_file)
    # time.sleep(5)
    # print("test 2\t" + str(time.time()), file=print_file)
    # print(fuc_return_test())
    # print(type(fuc_return_test()))
    # print(fuc_return_test()[0])
    # info = {
    #     'id': [1, 2, 3, 4],
    #     'temperature': [10, 8, 7, 9]
    # }
    # print(info['temperature'][info['id'][0]])
    # id temperature pmv ac door wd
    # a = [1, 8, 3, 0, 0, 1]
    # b = [2, 9, 2, 1, 0, 1]
    # c = [3, 7, 6, 1, 1, 1]
    # total = np.array([a, b, c])
    # print(total[:, 1])

    print(np.array([fuc_return_test()[0] for _ in range(4)]))
