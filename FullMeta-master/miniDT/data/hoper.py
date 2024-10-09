import pickle

path = r'./rainbow_dataset_15min_v2_256epoch.pkl'
# path = r'./hopper-expert-v2.pkl'

f = open(path, 'rb')
traj = pickle.load(f)

print(traj[0].keys())
#
# traj_length = len(traj)
#
# one_year_length = len(traj[0]['observations'])
#
# count_dict = {
#     "good": 0,
#     "ok": 0,
#     "bad": 0,
#     "uncount": 0
# }
#
# i = 0
#
# while i < traj_length:
#     j = 0
#     while j < one_year_length:
#         # state第一个是室外，第二个是室内
#         temper_out = traj[i]['observations'][j][0]
#         temper_in = traj[i]['observations'][j][1]
#
#         # action看第一个值，0是通风，1是制热，-1是制冷
#         action = traj[i]['actions'][j][0]
#
#         if 22.5 <= temper_in <= 26.5 and action == 0:
#             count_dict["good"] += 1
#         elif temper_out > 30 and action == -1:
#             count_dict["ok"] += 1
#         elif temper_out < 15 and action == 1:
#             count_dict["ok"] += 1
#         elif temper_in > 26.5 and action == 1:
#             count_dict["bad"] += 1
#         elif temper_in < 16.5 and action == -1:
#             count_dict["bad"] += 1
#         else:
#             count_dict["uncount"] += 1
#         j += 1
#         if j % 20 == 0:
#             print("Counted Year: {}, {}/{}".format(i, j, one_year_length))
#     i += 1
#
# print(count_dict)
# good = count_dict["good"]
# ok = count_dict["ok"]
# bad = count_dict["bad"]
# total = good + ok + bad
#
# print("Result:\n\tGood: {}, per: {:.2%}\n\tOk: {}, per: {:.2%}\n\tBad: {}, per: {:.2%}".format(good, good / total, ok,
#                                                                                                ok / total, bad,
#                                                                                                bad / total))
# uncount = count_dict["uncount"]
# all = total + uncount
#
# print("Statis: \n\tCounted: {}, per: {:.2%}\n\tUnCounted: {}, per: {:.2%}".format(total, total / all, uncount,
#                                                                                   uncount / all))
