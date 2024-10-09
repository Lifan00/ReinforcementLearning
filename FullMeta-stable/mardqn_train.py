import os
import pandas as pd
from scripts.multi_agent.MARDQN import train,test

train_time="202408160101"
def csv_total():
    # 设置包含CSV文件的目录
    directory = 'outputs/'
    # 设置输出文件的名称
    output_file = f'outputs/{train_time}/8agent.csv'
    output_file_directory = os.path.dirname(output_file)
    if not os.path.exists(output_file_directory):
        os.makedirs(output_file_directory, exist_ok=True)

    csv_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.csv')]
    df_list = []
    for file in csv_files:
        temp_df = pd.read_csv(file)
        df_list.append(temp_df)
    df = pd.concat(df_list, ignore_index=True)
    df.to_csv(output_file, index=False)

if __name__ == '__main__':
    Train=True
    if Train:
        train()
        checkpoint_dir = "scripts/data/tmp/" + train_time  # 模型地址 load_models(checkpoint_dir)
        test(checkpoint_dir)
    else:
        checkpoint_dir = "scripts/data/tmp/" + train_time  # 模型地址 load_models(checkpoint_dir)
        test(checkpoint_dir)
    csv_total()