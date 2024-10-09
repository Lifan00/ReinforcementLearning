import argparse
from miniDT.tqdm_train_backup import train

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # parser.add_argument("--local_rank", default=os.getenv("LOCAL_RANK", -1), type=int)
    # args = parser.parse_args()
    # train(args)
    train()
    # CUDA_VISIBLE_DEVICES=1 python dqn_train.py
    # tensorboard --logdir ./TMP
