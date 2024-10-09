import os

import pandas as pd

print(os.getcwd())
# 读取CSV文件
df = pd.read_csv(' ./hvac/data/CHN_Hubei.Wuhan.574940_CSWD.csv', encoding='utf-8')  # 请确保文件路径和编码方式正确

# 提取相对湿度列
humidity_data = df['相对湿度']

# 将相对湿度数据保存到TXT文件
with open('相对湿度数据.txt', 'w', encoding='utf-8') as file:
    for index, humidity in humidity_data.items():
        file.write(f'{humidity}\n')

print("相对湿度数据已提取到TXT文件。")