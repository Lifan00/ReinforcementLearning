import os

from scripts.DDPG import train

if __name__ == '__main__':
    os.chdir(os.getcwd())
    train()
