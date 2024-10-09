# SYSTEM
STATE_SPACE = 23
ACTION_SPACE = 11

# amplified
AMPL = 7

# ENERGY
AIR_DENSITY = 1.293
AIR_SHC = 1004

# WALL
# WALL_LAMBDA = 0.83  # 取自v8中的外墙参数，其相对应的内墙参数为0.9
WALL_LAMBDA = AMPL * 0.15  # 取自v8中的外墙参数，其相对应的内墙参数为0.9
WALL_THK = 0.3

# DOOR
DOOR_THK = 0.05  # 门的厚度貌似并不合理
DOOR_L = 2.3
DOOR_W = 0.8
# DOOR_LAMBDA = 2.504 * DOOR_THK  # 鉴于门窗的导热系数参数存在问题，这里进行相应的处理
DOOR_LAMBDA = AMPL * 0.043 * DOOR_THK  # 鉴于门窗的导热系数参数存在问题，这里进行相应的处理

# WINDOW
WINDOW_THK = 0.007
WINDOW_L = 1.3
WINDOW_W = 1.5
WINDOW_LAMBDA = AMPL * 1.09 * WINDOW_THK  # v8中窗的导热系数是6.17，该参数会决定空气的导热数值，暂且未定
# WINDOW_OPEN_AMPL = 3

# AIR
AIR_LAMBDA = AMPL * 9
# v8中空气的导热系数其实是正确的，应该很低，因为空气是绝缘的
# 但是，这里为了简化处理，认为开窗开门时，由空气进行传热，实际上传热效率应该很高

# ROOM
ROOM_H = 2.9
ROOM_L = 5
ROOM_W = 4
LR_L = 15
LR_W = 4

# AirConditioner
AC_LAMBDA = 5
AC_POWER = [-1, 1090, 0, 1220, -1]
AC_GENERATE = [-1, -3600, 0, 3960, -1]

# INITIALIZE
INIT_TEMPERATURE_IN = 15
INIT_TEMPERATURE_OUT = -1  # 初始值仍需定义
INIT_TIMESTAMP = 1104512400

# REWARD
REWARD_ALPHA = 0.5
REWARD_BETA = 0.85

TEMPERATURE_DEVIATION_LIMIT = 10  # 设定当前温度和最佳温度的偏差值最大为10度，超过10度将会截断
CALC_DEVIATION = 0

DEQUE_SIZE = 3

PMV_MAX = 3.5
PMV_MIN = 0
AC_MAX = 1

# PMV
# computePMV(TA, VEL, RH, MET, CLO, PA):
# TA = temperature air
# VEL = Relative air velocity
DEFAULT_VEL = 0.1
# RH = Relative humidity
DEFAULT_RH = 50
# MET = Metabolic rate
DEFAULT_MET = 1
# CLO = CLO_L[MONTH]
# PA = Partial water vapour pressure
DEFAULT_PA = 0
PMV_LIMIT = 7

# DATA_SPLIT
DATA_SPLIT_STEPS = 60
TIME_GAP = 3600 / DATA_SPLIT_STEPS

# PATH
# TEMPER_PATH = 'X:/Kode_Project/FullMeta/hvac/data/temper_out.txt'
# ILLUSION_PATH = 'X:/Kode_Project/FullMeta/hvac/data/illusion_out.txt'
# DICT_PATH = "X:/Kode_Project/FullMeta/hvac/data/that_dict.json"
TEMPER_PATH = './hvac/data/temper_out.txt'
ILLUSION_PATH = './hvac/data/illusion_out.txt'
DICT_PATH = "./hvac/data/that_dict.json"

# kWh2Joule ratio
RATIO_KWH2JOULE = 3600000
