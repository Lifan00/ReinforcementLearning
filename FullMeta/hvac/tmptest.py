# simulation 测试
# arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# size = len(arr)
# index = 0
# while True:
#     if index < size:
#         print(index)
#         print(size)
#         print(arr[index])
#         index += 1
#
#     else:
#         print(index)
#         print(size)
#         break
#
# print("done")
from statistics import mean

# 动作空间测试
# action = [
#     1, 2, 3,
#     4, 5, 6,
#     7, 8, 9,
#     10, 11
# ]
#
# a0 = action[-2:]
# a1 = action[0:3]
# a2 = action[3:6]
# a3 = action[6:9]
# print(a0, a1, a2, a3)
#
# outer = [-2, -1, 0, 1, 2]
# inner = [0, 1, 2, 3, 4]

# PMV
# 计算测试
# TA = 21.9
# month = 12
# pmv = computePMV(TA, DEFAULT_VEL, DEFAULT_RH, DEFAULT_MET, CLO_L[month], DEFAULT_PA)
# print(pmv)
# print(air_condition_restrictor(0, 14))

# def append_element(power_list):
#     min_power = min(power_list)
#     max_power = max(power_list)
#     normalized_power_list = [normalize(power, min_power, max_power) for power in power_list]
#     print(normalized_power_list)
#     return mean(normalized_power_list)
#
#
# original_list = [1090, 0, 1220, 1090, 1220, 1220, 1090, 1220, 1220]
# means = append_element(original_list)
#
# print(original_list)  # 输出 [1, 2, 3, 4]
# print(means)


# from collections import deque
#
# # 创建一个长度为15的队列
# my_queue = deque(maxlen=5)
#
# # 向队列中添加元素，使用 append() 方法
# my_queue.append(1)
# my_queue.append(2)
# my_queue.append(3)
# print(len(my_queue))
# my_queue.append(4)
# my_queue.append(5)
# my_queue.append(6)
#
# # 当队列长度达到15时，新元素将自动从队列的头部弹出
# print(my_queue)  # 输出: deque([1, 2, 3], maxlen=15)
#
# my_queue.append(4)  # 新元素添加到队尾
# print(my_queue)  # 输出: deque([2, 3, 4], maxlen=15)
# print(len(my_queue))
# my_queue.clear()
# print(my_queue)  #
# print(len(my_queue))


# num_updates_per_iter = 200
# with trange(num_updates_per_iter) as tqdm_iter:
#     print("?")
#     for _ in tqdm_iter:
#         print(_)


# for i in trange(4, desc='1st loop'):
#     for j in trange(5, desc='2nd loop'):
#         for k in trange(50, desc='3rd loop', leave=False):
#             sleep(0.01)


arr = [-1, -2, -1, 2, 3]
a = mean([abs(x) for x in arr])
print(a)
