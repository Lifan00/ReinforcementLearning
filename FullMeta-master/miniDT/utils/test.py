import argparse
import os
import pickle

import numpy as np
import torch
from tqdm import tqdm

from hvac.entity.EnvCore import EnvCore
from hvac.tools.scripts import DataRecorder
from miniDT.utils.config import RTG_SCALE, CONTEXT_LEN, N_BLOCKS, EMBED_DIM, N_HEADS, DROPOUT_PROB, STATE_DIM, \
    ACTION_DIM
from miniDT.utils.model import DecisionTransformer


def evaluate_on_env(model, device, context_len, rtg_target, rtg_scale=RTG_SCALE,
                    num_eval_ep=1, max_test_ep_len_year_len=525541,
                    state_mean=None, state_std=None):
    """
    context_len 和模型相关，是上下文长度
    num_eval_ep 是测试的轮次
    max_test_ep_len 是一年的长度，默认是525541
    """
    eval_batch_size = 1  # required for forward pass

    total_reward = 0

    state_dim = STATE_DIM
    act_dim = ACTION_DIM

    if state_mean is None:
        state_mean = torch.zeros((state_dim,)).to(device)
    else:
        state_mean = torch.from_numpy(state_mean).to(device)

    if state_std is None:
        state_std = torch.ones((state_dim,)).to(device)
    else:
        state_std = torch.from_numpy(state_std).to(device)

    # same as timesteps used for training the transformer
    # also, crashes if device is passed to arange()
    timesteps = torch.arange(start=0, end=max_test_ep_len_year_len, step=1)
    timesteps = timesteps.repeat(eval_batch_size, 1).to(device)

    env = EnvCore()
    model.eval()

    name = "Decision_Transformer_Output_Info"
    dr = DataRecorder(name)

    with torch.no_grad():

        for _ in range(num_eval_ep):

            # zeros place holders
            actions = torch.zeros((eval_batch_size, max_test_ep_len_year_len, act_dim),
                                  dtype=torch.float32, device=device)
            states = torch.zeros((eval_batch_size, max_test_ep_len_year_len, state_dim),
                                 dtype=torch.float32, device=device)
            rewards_to_go = torch.zeros((eval_batch_size, max_test_ep_len_year_len, 1),
                                        dtype=torch.float32, device=device)

            # init episode
            running_state = env.reset()
            running_reward = 0
            running_rtg = rtg_target / rtg_scale

            dr.start(True)

            for i in tqdm(range(max_test_ep_len_year_len)):
                t = i // 15
                if i % 15 == 0:
                    # add state in placeholder and normalize
                    states[0, t] = torch.from_numpy(np.array(running_state)).to(device)
                    states[0, t] = (states[0, t] - state_mean) / state_std

                    # calculate running rtg and add it in placeholder
                    running_rtg = running_rtg - (running_reward / rtg_scale)
                    rewards_to_go[0, t] = running_rtg

                    if t < context_len:
                        _, act_preds, _ = model.forward(timesteps[:, :context_len],
                                                        states[:, :context_len],
                                                        actions[:, :context_len],
                                                        rewards_to_go[:, :context_len])
                        act = act_preds[0, t].detach()
                    else:
                        _, act_preds, _ = model.forward(timesteps[:, t - context_len + 1:t + 1],
                                                        states[:, t - context_len + 1:t + 1],
                                                        actions[:, t - context_len + 1:t + 1],
                                                        rewards_to_go[:, t - context_len + 1:t + 1])
                        act = act_preds[0, -1].detach()

                    filtered_action = action_filter(act.cpu().numpy())
                    running_state, running_reward, done, _ = env.step(filtered_action.astype(int).tolist())
                    if done:
                        break
                    dr.collect(env.get_info()[0], running_reward)

                    # add action in placeholder
                    actions[0, t] = act

                    total_reward += running_reward

                    if done:
                        break
                else:
                    if done:
                        break
                    else:
                        running_state, running_reward, done, _ = env.step(filtered_action.astype(int).tolist())
            dr.end()
            avg_pmv_people_exist = dr.print()[0]

    return total_reward / num_eval_ep, avg_pmv_people_exist


def action_filter(action):
    # ac1 wd1 door1
    # ac2 wd2 door2
    # ac3 wd2 door3
    # ac4 door4
    for i in range(0, len(action), 3):
        if action[i] < -0.5:
            action[i] = -1
        elif action[i] > 0.5:
            action[i] = 1
        else:
            action[i] = 0
    for i in range(1, len(action), 3):
        action[i] = 1 if action[i] > 0.5 else 0
    for i in range(2, len(action), 3):
        action[i] = 1 if action[i] > 0.5 else 0
    return action


def count_target(path='./perfect_v3/test_perfect_data_v3_20231114153426.pkl'):
    f = open(path, 'rb')
    trajectories = pickle.load(f)

    trajectories_len = len(trajectories)

    reward_list = []

    for i in range(trajectories_len):
        reward_list.append(sum(trajectories[i]['rewards']))

    return sum(reward_list) / len(reward_list), reward_list


def test(args):
    context_len = CONTEXT_LEN  # K in decision transformer
    n_blocks = N_BLOCKS  # num of transformer blocks
    embed_dim = EMBED_DIM  # embedding (hidden) dim of transformer
    n_heads = N_HEADS  # num of transformer heads
    dropout_p = DROPOUT_PROB  # dropout probability

    eval_chk_pt_dir = args.chk_pt_dir
    eval_chk_pt_name = args.chk_pt_name
    eval_chk_pt_list = [eval_chk_pt_name]

    device = torch.device(args.device)
    print("device set to: ", device)

    # eval_env = gym.make(eval_env_name)

    state_dim = STATE_DIM
    act_dim = ACTION_DIM

    for eval_chk_pt_name in eval_chk_pt_list:
        eval_model = DecisionTransformer(
            state_dim=state_dim,
            act_dim=act_dim,
            n_blocks=n_blocks,
            h_dim=embed_dim,
            context_len=context_len,
            n_heads=n_heads,
            drop_p=dropout_p,
        ).to(device)

        eval_chk_pt_path = os.path.join(eval_chk_pt_dir, eval_chk_pt_name)

        # load checkpoint
        eval_model.load_state_dict(torch.load(eval_chk_pt_path, map_location=device))

        print("model loaded from: " + eval_chk_pt_path)

        # evaluate on env
        results = evaluate_on_env(eval_model, device, context_len, rtg_target=30000)
        print(results)

    print("total num of checkpoints evaluated: " + str(len(eval_chk_pt_list)))
    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--local_rank", type=int, default=os.getenv("LOCAL_RANK", -1))
    parser.add_argument("--device_ids", type=str, default='0,1', help="Training Devices, example: '0,1,2'")
    parser.add_argument('--chk_pt_dir', type=str, default='dt_runs/')
    parser.add_argument('--chk_pt_name', type=str,
                        default='dt_info_tmptest_perfect_data_v3_20231114153426v3_batch64_context60_epoch400_update500_HEADx8_BLOCKS4_model_23-11-15-21-24-48.pt')
    parser.add_argument('--device', type=str, default='cuda')
    conf = parser.parse_args()
    test(conf)
    # python
    # dp_train.py - -device_ids = 0, 1
    # CUDA_VISIBLE_DEVICES = 1
    # python
    # dqn_train.py
    # tensorboard - -logdir. / TMP
