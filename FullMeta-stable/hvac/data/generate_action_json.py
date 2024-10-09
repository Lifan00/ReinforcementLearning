"""
1   空调  2   窗户  3   门
4   空调  5   窗户  6   门
7   空调  8   窗户  9   门
10  空调  11  窗户
目前空调档位是-1,0,1
"""
import json


def rational(a, b, c, d):
    tmp_list = [a, b, c, d]
    lmax = max(tmp_list)
    lmin = min(tmp_list)
    return lmax - lmin != 2


action_dict = {}
index = 0

for air0 in range(-1, 2):
    for air1 in range(-1, 2):
        for air2 in range(-1, 2):
            for air3 in range(-1, 2):
                is_rational = rational(air0, air1, air2, air3)
                if is_rational:
                    window0 = abs(air0)
                    window1 = abs(air1)
                    window2 = abs(air2)
                    window3 = abs(air3)
                    for door1 in range(2):
                        for door2 in range(2):
                            for door3 in range(2):
                                action_dict[index] = [
                                    air1, window1, door1,
                                    air2, window2, door2,
                                    air3, window3, door3,
                                    air0, window0
                                ]
                                index += 1
print(action_dict)
print(len(action_dict))

json_file = open("that_dict.json", "w")
json.dump(action_dict, json_file)
json_file.close()
# 248
