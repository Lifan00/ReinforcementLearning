from hvac.tools.scripts import PklRecorder

# action_preds = tensor([[0.1063, 0.2050, 0.2185, 0.2209, 0.0909, 0.1912],
#                        [0.0224, 0.1197, 0.1345, 0.1314, 0.0359, 0.0995],
#                        [0.0715, 0.0462, 0.0492, 0.0955, 0.1039, 0.1073]],
#                       device='cuda:0')
# action_target = tensor([[0., 0., 0., 0., 0., 0.],
#                         [1., 1., 1., 1., 1., 1.],
#                         [0., 0., 0., 0., 0., 0.]], device='cuda:0')
#
# try_dict = {
#     "0": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     "1": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#     "2": [-1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1]
# }
#
# action_preds1 = tensor([[-1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1],
#                         [-1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1]], device='cuda:0')
# action_target1 = tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], device='cuda:0')
#
# action_loss = F.mse_loss(action_preds1.float(), action_target1.float(), reduction='mean')
#
#
# def clac_loss(action_preds, action_target):
#     calc_loss = 0
#     for i, row in enumerate(action_preds):
#         for j, value in enumerate(row):
#             loss = (action_preds[i, j] - action_target[i, j]) ** 2
#             if j == 0 or j == 3 or j == 6 or j == 9:
#                 loss *= 2
#             calc_loss += loss
#     return calc_loss / action_preds.numel()
#
#
# print(clac_loss(action_preds1, action_target1))
#
# print(action_loss)
# pkllist = []
#
#
# def func(i):
#     pkllist.append(i)
#
#
# EPISODES = 200
# limit = 64
#
# with trange(EPISODES) as tqdm_episodes:
#     for i, _ in enumerate(tqdm_episodes):
#         if i >= EPISODES - limit:
#             func(i)
#
# print(len(pkllist))

pr = PklRecorder()

pr.data = [1, 2, 3, 4, 5, 6, 7]
pr.save("test")
