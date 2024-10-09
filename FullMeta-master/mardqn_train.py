import time
from scripts.multi_agent.MARDQN import  train

if __name__ == '__main__':
    start_time = time.time()
    train()
    end_time = time.time()
    run_time = end_time - start_time
    print("main函数运行时间：", run_time)
