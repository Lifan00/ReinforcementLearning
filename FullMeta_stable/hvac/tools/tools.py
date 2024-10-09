import csv

import math
import time
from statistics import mean

from hvac.tools.config import DATA_SPLIT_STEPS, TIME_GAP, TEMPERATURE_DEVIATION_LIMIT, REWARD_ALPHA, REWARD_BETA, \
    CALC_DEVIATION, TEMPER_PATH, ILLUSION_PATH


def air_condition_restrictor(action, temperature):
    return 0 if (action > 0 and temperature > 27) or (action < 0 and temperature < 17) else action


def normalize(value, min_value, max_value):
    if max_value == min_value:
        return 0 if max_value == 0 else 1
    return (value - min_value) / (max_value - min_value)


def calculate_reward(power_list, temperature, absed_pmv, people_exist, time_stamp):
    use_new = False
    if use_new:
        min_power = min(power_list)
        max_power = max(power_list)
        normalized_power_list = [normalize(power, min_power, max_power) for power in power_list]
        normalized_power_part = mean(normalized_power_list)

        # 通过计算 temperature 到 pmv=0  时的最佳温度偏差来将温度归一化到 0-1
        month = stamp_2_month(time_stamp)
        best_temperature = TEMPERATURE_BEST_PMV0[month]
        deviation = temperature - best_temperature
        deviation = max(-TEMPERATURE_DEVIATION_LIMIT, min(TEMPERATURE_DEVIATION_LIMIT, deviation))  # 截断
        normalized_temperature_deviation_part = normalize(deviation, -TEMPERATURE_DEVIATION_LIMIT,
                                                          TEMPERATURE_DEVIATION_LIMIT)

        normalized_pmv_part = normalize(absed_pmv, 0, 7)

        GAMMA = REWARD_ALPHA if people_exist else REWARD_BETA
        x = GAMMA * normalized_power_part + (
                1 - GAMMA) * normalized_pmv_part + CALC_DEVIATION * normalized_temperature_deviation_part
        return 1 / x, [normalized_power_part, normalized_temperature_deviation_part, normalized_pmv_part]
    else:
        reward = (7 - absed_pmv) / 7 if people_exist else ((1220 - (mean(power_list) + 0.05)) / 1220) / 2
        return reward, []


def data_split(data_arr, steps=DATA_SPLIT_STEPS):  # DATA_SPLIT_STEPS为步长，，原始数据是一小时一次检测数据，比如n为60则时间划分为60份，n为15则分为15份
    array_len = len(data_arr)
    data = [data_arr[0]]
    for i in range(array_len - 1):
        gap = data_arr[i + 1] - data_arr[i]
        step_size = gap / steps
        for j in range(steps):
            data.append(data_arr[i] + step_size * (j + 1))
    return data


def import_temper():  # 引入室外温度
    with open(TEMPER_PATH, 'r') as f:
        file = f.readlines()  # txt中所有字符串读入data
        data = []
        for line in file:
            odom = line.split()  # 将单个数据分隔开存好
            numbers_float = map(float, odom)  # 转化为浮点数
            for i in numbers_float:
                data.append(i)
    return data


def import_illu():  # 引入室外天顶亮度
    with open(ILLUSION_PATH, 'r') as f:
        file = f.readlines()  # txt中所有字符串读入data
        data = []
        for line in file:
            odom = line.split()  # 将单个数据分隔开存好
            numbers_float = map(float, odom)  # 转化为浮点数
            for i in numbers_float:
                data.append(i)
    return data


def calc_heat_energy(gamma, A, temperature_d, thk):
    """
    Calculate the heat energy of a gas.

    Args:
        gamma (float): thermal conductivity in W/mK
        A (float): conduct area
        temperature_d (float): temperature difference in degrees Celsius
        thk (float): conduct thickness

    Returns: heat energy
    """
    return TIME_GAP * gamma * A * temperature_d / thk


def stamp_2_month(ts):
    dt = time.strftime("%m", time.localtime(int(float(ts))))
    return int(dt)


def stamp_2_hour(ts):
    dt = time.strftime("%H", time.localtime(int(float(ts))))
    return int(dt)


def convert_json_key(param_dict):
    """
    json.dump不支持key是int的dict，在编码存储的时候会把所有的int型key写成str类型的
    所以在读取json文件后，用本方法将所有的被解码成str的int型key还原成int
    """
    new_dict = dict()
    for key, value in param_dict.items():
        # noinspection PyBroadException
        try:
            new_key = int(key)
            new_dict[new_key] = value
        except:
            new_dict[key] = value

    return new_dict


CLO_L = [
    0,
    1.34,
    1.18,
    0.83,
    0.59,
    0.41,
    0.33,
    0.31,
    0.31,
    0.44,
    0.51,
    0.76,
    1.26
]

TEMPERATURE_BEST_PMV0 = [
    0,
    21.4,
    22.3,
    24.2,
    25.6,
    26.6,
    27.0,
    27.1,
    27.1,
    26.4,
    26.0,
    24.6,
    21.9
]


# Take temperature T in C
# Return saturated vapour pressure, in kPa
def FNPS(T):
    # Note: Missing '(' in document
    return math.exp(16.6536 - 4030.183 / (T + 235.0))


# Clothing, clo,      穿衣指数                CLO
# Metabolic rate, met,    代谢率           MET
# External work, met,   外部工作             WME
# Air temperature, C,       空气温度         TA
# Mean radiant temperature, C,  平均辐射温度    TR
# Relative air velocity, m/s,  相对气流速度      VEL
# Relative humidity, %,  相对湿度            RH
# Partial water vapour pressure, Pa,PA      部分水汽压,直接设置为0
def computePMV(TA, VEL, RH, MET, CLO, PA):
    if PA == 0:
        PA = RH * 10 * FNPS(TA)  # water vapour pressure, Pa
    ICL = 0.155 * CLO  # thermal insulation of the clothing in m2K/W
    M = MET * 58.15  # external work in W/m2
    TR = TA
    #   W = WME * 58.15
    MW = M  # internal heat production in the human body
    if ICL <= 0.078:
        FCL = 1 + 1.29 * ICL
    else:
        FCL = 1.05 + 0.645 * ICL  # clothing area factor
    HCF = 12.1 * math.sqrt(VEL)  # heat transf. coeff. by forced convection
    TAA = TA + 273  # air temperature in Kelvin
    TRA = TR + 273  # mean radiant temperature in Kelvin

    TCLA = TAA + (35.5 - TA) / (3.5 * ICL + 0.1)  # first guess for surface temperature of clothing
    P1 = ICL * FCL
    P2 = P1 * 3.96
    P3 = P1 * 100
    P4 = P1 * TAA
    # Note: P5 = 308.7 - 0.028 * MW + P2 * (TRA / 100) * 4  in document
    P5 = (308.7 - 0.028 * MW) + (P2 * math.pow(TRA / 100, 4))
    # Note: TLCA in document
    XN = TCLA / 100
    # Note: XF = XN in document
    XF = TCLA / 50
    N = 0  # number of iterations
    EPS = 0.00015  # stop criteria in iteration
    # Note: HC must be defined before use
    HC = HCF

    while abs(XN - XF) > EPS:
        XF = (XF + XN) / 2
        HCN = 2.38 * math.pow(abs(100.0 * XF - TAA), 0.25)
        if HCF > HCN:
            HC = HCF
        else:
            HC = HCN
        # Note: should be '-' in document
        XN = (P5 + P4 * HC - P2 * math.pow(XF, 4)) / (100 + P3 * HC)
        N = N + 1
        if N > 150:
            print('Max iterations exceeded')
            return 999999
    TCL = 100 * XN - 273

    HL1 = 3.05 * 0.001 * (5733 - 6.99 * MW - PA)  # heat loss diff. through skin
    if MW > 58.15:
        HL2 = 0.42 * (MW - 58.15)
    else:
        HL2 = 0
    HL3 = 1.7 * 0.00001 * M * (5867 - PA)  # latent respiration heat loss
    HL4 = 0.0014 * M * (34 - TA)  # dry respiration heat loss
    # Note: HL5 = 3.96 * FCL * (XN^4 - (TRA/100^4)   in document
    HL5 = 3.96 * FCL * (math.pow(XN, 4) - math.pow(TRA / 100, 4))  # heat loss by radiation
    HL6 = FCL * HC * (TCL - TA)

    TS = 0.303 * math.exp(-0.036 * M) + 0.028
    PMV = TS * (MW - HL1 - HL2 - HL3 - HL4 - HL5 - HL6)
    # PMV = abs(PMV)
    # PMV=10-PMV
    return PMV  # PMV越小越好


