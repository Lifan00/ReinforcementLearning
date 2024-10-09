import pickle

from tqdm import tqdm

path = r'./splittmptest_perfect_data_v2_20231123215003v3.pkl'
f = open(path, 'rb')
traj = pickle.load(f)

traj_length = len(traj)
one_month_length = len(traj[0]['observations'])

TEMPERATURE_BEST_PMV0 = [
    21.9,  # 这个是12月，对应12%12=0的计算
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
show_diverse_count = True

# diverse_dict 用来存储信息
action0_diverse_dict = {}
action1_diverse_dict = {}
action2_diverse_dict = {}

# 0是good, 1是bad, 2是3度范围内通风
action0_count_list = [0, 0, 0]

# 0是good, 1是bad
action1_count_list = [0, 0]
action2_count_list = [0, 0]

with tqdm(range(traj_length)) as tqdm_trajs:
    for i_month in tqdm_trajs:
        best_temperature = TEMPERATURE_BEST_PMV0[(1 + i_month) % 12]
        tqdm_trajs.set_description(f"Month: {12 if (1 + i_month) % 12 == 0 else (1 + i_month) % 12:02d}", True)
        for j_index in range(one_month_length):
            # state第一个是室外，第二个是室内
            temper_out = traj[i_month]['observations'][j_index][0]
            temper_in = traj[i_month]['observations'][j_index][1]

            # action看第一个值，0是通风，1是制热，-1是制冷
            action = traj[i_month]['actions'][j_index][0]

            # diverse > 0 means that now is too hot and needs cold
            diverse = temper_in - best_temperature

            if action == 0:  # 测试通风时的
                # 通过以下代码确定，在正负3度左右时，会集中开窗
                # if abs(int_abs_diverse) > 3:
                #     print(f"Diverse: {int_abs_diverse}, Month: {(1 + i_month) % 12}, In: {temper_in}, Out: {temper_out}")
                action0_diverse_dict[int(diverse)] = action0_diverse_dict.setdefault(int(diverse), 0) + 1

                if diverse > 0 and temper_in > temper_out:
                    action0_count_list[0] += 1
                elif diverse < 0 and temper_in < temper_out:
                    action0_count_list[0] += 1
                elif abs(diverse) < 4:  # 预计是波动4度左右是可以接受的范围
                    action0_count_list[2] += 1
                else:
                    print(f"IN:{temper_in:.2f}\tOUT:{temper_out:.2f}\tDIVERSE:{diverse:.2f}")
                    action0_count_list[1] += 1
            elif action == 1:  # 加热情况
                action1_diverse_dict[int(diverse)] = action1_diverse_dict.setdefault(int(diverse), 0) + 1
                action1_count_list[0 if diverse < 0 else 1] += 1
            else:  # 制冷情况
                action2_diverse_dict[int(diverse)] = action2_diverse_dict.setdefault(int(diverse), 0) + 1
                action2_count_list[0 if diverse > 0 else 1] += 1

if show_diverse_count:
    print(dict(sorted(action0_diverse_dict.items())))
    print(dict(sorted(action1_diverse_dict.items())))
    print(dict(sorted(action2_diverse_dict.items())))

print(f"\n=====总统计:{sum(action0_count_list):,}=====")
print(f"通风情况\t好：{action0_count_list[0]:,} ({action0_count_list[0] / sum(action0_count_list):.2%})")
print(f"通风情况\t坏：{action0_count_list[1]:,} ({action0_count_list[1] / sum(action0_count_list):.2%})")
print(f"通风情况\t允许范围内：{action0_count_list[2]:,} ({action0_count_list[2] / sum(action0_count_list):.2%})")

print(f"\n=====总统计:{sum(action1_count_list):,}=====")
print(f"加热情况\t好：{action1_count_list[0]:,} ({action1_count_list[0] / sum(action1_count_list):.2%})")
print(f"加热情况\t坏：{action1_count_list[1]:,} ({action1_count_list[1] / sum(action1_count_list):.2%})")

print(f"\n=====总统计:{sum(action2_count_list):,}=====")
print(f"制冷情况\t好：{action2_count_list[0]:,} ({action2_count_list[0] / sum(action2_count_list):.2%})")
print(f"制冷情况\t坏：{action2_count_list[1]:,} ({action2_count_list[1] / sum(action2_count_list):.2%})")

# 计算汇总
total = sum(action0_count_list) + sum(action1_count_list) + sum(action2_count_list)
good = action0_count_list[0] + action0_count_list[2] + action1_count_list[0] + action2_count_list[0]
bad = action0_count_list[1] + action1_count_list[1] + action2_count_list[1]
print(f"\n=====汇总:{total:,}=====")
print(f"好：{good:,} ({good / total:.2%})")
print(f"坏：{bad:,} ({bad / total:.2%})")
