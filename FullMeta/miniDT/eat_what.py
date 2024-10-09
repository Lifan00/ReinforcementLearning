import random
import time

food_list = ['大碗菜', '兰州拉面', '三楼', '馒头', '烧卤饭',
             '卤菜', '新农面', '本土菜', '烤肉饭', '重庆小面',
             '渔粉', '农家菜', '新疆菜']

print("=======菜单=======")
for i in range(len(food_list)):
    print("-", str(i + 1).zfill(2), food_list[i])
print("- xx 奖励自己")
print("=======菜单=======")
rand_food_number_seed = int(input("输入一个1-" + str(len(food_list)) + "的幸运号码: "))
random.seed(rand_food_number_seed)
spur = random.random()
if spur > 0.07:
    any_number = (random.random() * time.time() / random.random()) ** rand_food_number_seed
    result = int(any_number % len(food_list))
    print("今天我要吃", food_list[result] + "!")
else:
    print("是时候 奖励自己了！")
