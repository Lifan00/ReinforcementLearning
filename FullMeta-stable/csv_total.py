import pandas as pd
import os

# 设置包含CSV文件的目录
directory = 'outputs/'
# 设置输出文件的名称
output_file = 'outputs/202408023008/8agent.csv'

# 获取目录中所有的CSV文件
csv_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.csv')]

# 创建一个空的DataFrame，用于存储合并后的数据
df_list = []

# 遍历所有CSV文件
for file in csv_files:
    # 读取CSV文件
    temp_df = pd.read_csv(file)
    # 将读取的DataFrame添加到列表中
    df_list.append(temp_df)

# 使用concat合并所有DataFrame
df = pd.concat(df_list, ignore_index=True)

# 将合并后的DataFrame保存为新的CSV文件
df.to_csv(output_file, index=False)
print(f'Files have been merged into {output_file}')