import argparse
import csv
import json
import os.path
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# from torch.utils.data.distributed import DistributedSampler
from tqdm import trange

from utils.config import *
from utils.model import DecisionTransformer
from utils.tools import TrajectoryDataset, evaluate_on_env, get_rl_normalized_score

def train(args=None):
    dataset_name = DATASET_NAME
    rtg_scale = RTG_SCALE

    batch_size = BATCH_SIZE
    lr = LEARN_RATE
    wt_decay = WEIGHT_DECAY
    warmup_steps = WARMUP_STEPS

    max_train_iters = MAX_ITERS
    num_updates_per_iter = UPDATES_PER_ITER

    context_len = CONTEXT_LEN  # K in decision transformer
    n_blocks = N_BLOCKS  # num of transformer blocks
    embed_dim = EMBED_DIM  # embedding (hidden) dim of transformer
    n_heads = N_HEADS  # num of transformer heads
    dropout_p = DROPOUT_PROB  # dropout probability

    state_dim = STATE_DIM
    act_dim = ACTION_DIM

    dataset_path = f'./data/{dataset_name}.pkl'
    log_dir = "outputs/dt_runs/"
    tensor_dir = "outputs/tensor_logs/"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # training and evaluation device
    device_ids = list(map(int, args.device_ids.split(',')))  # device:3,4,5,6 -> rank [0,1,2,3]
    dist.init_process_group(backend='nccl')
    device = torch.device("cuda:{}".format(device_ids[args.local_rank]))
    torch.cuda.set_device(device)

    start_time = datetime.now().replace(microsecond=0)
    start_time_str = start_time.strftime("%y-%m-%d-%H-%M-%S")

    prefix = "dt_" + dataset_name

    save_model_name = prefix + "_model_" + start_time_str + ".pt"
    save_model_path = os.path.join(log_dir, save_model_name)
    save_best_model_path = save_model_path[:-3] + "_best.pt"

    log_csv_name = prefix + "_log_" + start_time_str + ".csv"
    log_csv_path = os.path.join(log_dir, log_csv_name)

    csv_writer = csv.writer(open(log_csv_path, 'a', 1))
    csv_header = (["duration", "num_updates", "action_loss",
                   "eval_avg_reward", "eval_avg_ep_len", "eval_rl_score"])
    csv_writer.writerow(csv_header)
    tensor_writer = SummaryWriter(tensor_dir)

    print("=" * 60)
    print("start time: " + start_time_str)
    print("=" * 60)

    print("device set to: " + str(device))
    print("dataset path: " + dataset_path)
    print("model save path: " + save_model_path)
    print("log csv save path: " + log_csv_path)
    print("=" * 60)

    traj_dataset = TrajectoryDataset(dataset_path, context_len, rtg_scale)

    # traj_datat_sampler = DistributedSampler(traj_dataset)

    traj_data_loader = DataLoader(
        traj_dataset,
        batch_size=batch_size,
        # sampler=traj_datat_sampler,
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )

    data_iter = iter(traj_data_loader)

    # get state stats from dataset
    # state_mean, state_std = traj_dataset.get_state_stats()

    model = DecisionTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        n_blocks=n_blocks,
        h_dim=embed_dim,
        context_len=context_len,
        n_heads=n_heads,
        drop_p=dropout_p,
    ).to(device)

    model = DistributedDataParallel(model,
                                    device_ids=[device_ids[args.local_rank]],
                                    output_device=device_ids[args.local_rank],
                                    find_unused_parameters=True)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=wt_decay
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps + 1) / warmup_steps, 1)
    )

    max_rl_score = -1.0
    total_updates: int = 0
    info_csv = {"actions": [], "steps": []}
    info_name = "info_" + dataset_name + "_batch" + str(batch_size) + "_" + "context" + str(context_len)

    for i_train_iter in range(max_train_iters):

        info_csv = {"actions": [], "steps": []}
        log_action_losses = []
        model.train()

        with trange(num_updates_per_iter) as tqdm_iter:
            for _ in tqdm_iter:
                tqdm_iter.set_description("Epoch %i / %i" % (i_train_iter + 1, max_train_iters))
                try:
                    timesteps, states, actions, returns_to_go, traj_mask = next(data_iter)
                except StopIteration:
                    data_iter = iter(traj_data_loader)
                    timesteps, states, actions, returns_to_go, traj_mask = next(data_iter)

                timesteps = timesteps.to(device)  # B x T
                states = states.to(device)  # B x T x state_dim
                actions = actions.to(device)  # B x T x act_dim
                returns_to_go = returns_to_go.to(device).unsqueeze(dim=-1)  # B x T x 1
                traj_mask = traj_mask.to(device)  # B x T
                action_target = torch.clone(actions).detach().to(device)

                state_preds, action_preds, return_preds = model.forward(
                    timesteps=timesteps,
                    states=states,
                    actions=actions,
                    returns_to_go=returns_to_go
                )
                # only consider non padded elements
                action_preds = action_preds.view(-1, act_dim)[traj_mask.view(-1, ) > 0]
                action_target = action_target.view(-1, act_dim)[traj_mask.view(-1, ) > 0]

                action_loss = F.mse_loss(action_preds, action_target, reduction='mean')

                optimizer.zero_grad()
                action_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
                optimizer.step()
                scheduler.step()

                log_action_losses.append(action_loss.detach().cpu().item())

                info_csv["actions"].append([action_preds.detach().cpu().numpy().tolist()])
                info_csv["steps"].append([timesteps.tolist()])

        # **********************************************************************
        # evaluate action accuracy
        results = evaluate_on_env()
        eval_avg_reward = results['eval/avg_reward']
        eval_avg_ep_len = results['eval/avg_ep_len']
        eval_rl_score = get_rl_normalized_score(results['eval/avg_reward']) * 100
        # **********************************************************************

        # print and save csv, tensor,
        if args.local_rank == 0:
            mean_action_loss = np.mean(log_action_losses)
            time_elapsed = str(datetime.now().replace(microsecond=0) - start_time)
            total_updates += num_updates_per_iter

            log_str = (
                    "time elapsed: " + time_elapsed + '\n' +
                    "num of updates: " + str(total_updates) + '\n' +
                    "action loss: " + format(mean_action_loss, ".5f") + '\n' +
                    "eval avg reward: " + format(eval_avg_reward, ".5f") + '\n' +
                    "eval avg ep len: " + format(eval_avg_ep_len, ".5f") + '\n' +
                    "eval rl_one score: " + format(eval_rl_score, ".5f"))
            print(log_str)
            log_data = [time_elapsed, total_updates, mean_action_loss,
                        eval_avg_reward, eval_avg_ep_len,
                        eval_rl_score]
            csv_writer.writerow(log_data)
            tensor_writer.add_scalar("action_mean_loss", mean_action_loss, total_updates)

        # save model
        if args.local_rank == 0:
            print("max rl_one score: " + format(max_rl_score, ".5f"))
            if eval_rl_score >= max_rl_score:
                print("saving max rl_one score model at: " + save_best_model_path)
                torch.save(model.state_dict(), save_best_model_path)
                max_rl_score = eval_rl_score

            print("saving current model at: " + save_model_path)
            torch.save(model.state_dict(), save_model_path)
            print("=" * 60)

    tensor_writer.close()
    print("=" * 60)
    print("finished training!")
    print("=" * 60)
    end_time = datetime.now().replace(microsecond=0)
    time_elapsed = str(end_time - start_time)
    end_time_str = end_time.strftime("%y-%m-%d-%H-%M-%S")
    print("started training at: " + start_time_str)
    print("finished training at: " + end_time_str)
    print("total training time: " + time_elapsed)
    print("max rl_one score: " + format(max_rl_score, ".5f"))
    print("saved max rl_one score model at: " + save_best_model_path)
    print("saved last updated model at: " + save_model_path)
    print("=" * 60)

    f = open("outputs/" + info_name + ".json", "w")
    json.dump(info_csv, f)
    f.close()
    print("save info successfully")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DT training")

    parser.add_argument("--local_rank", type=int, default=-1, help="DPP parameter, do not modify")
    parser.add_argument("--device_ids", type=str, default='0', help="Training Devices, example: '0,1,2'")
    conf = parser.parse_args()
    train(conf)
    # python -m torch.distributed.launch --nproc_per_node 2 --master_port 4444 dqn_train.py --device_ids=0,1
    # about port: netstat -ntlp | grep 4444
    # tensorboard --logdir ./TMP
