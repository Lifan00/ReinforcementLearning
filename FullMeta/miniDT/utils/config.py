# **********************************************************************
STATE_DIM = 23
ACTION_DIM = 11

DATASET_NAME = "tmptest_perfect_data_v3_20231114153426v3"
#DATASET_NAME = "hopper-medium-replay-v2"

RTG_SCALE = 500000  # normalize returns to go

BATCH_SIZE = 64
LEARN_RATE = 1e-4
WEIGHT_DECAY = 1e-4
WARMUP_STEPS = 10000  # warmup steps for lr scheduler

# total updates = max_train_iters x num_updates_per_iter
MAX_ITERS = 400
UPDATES_PER_ITER = 500

CONTEXT_LEN = 60  # K in decision transformer
N_BLOCKS = 4  # num of transformer blocks
EMBED_DIM = 128  # embedding (hidden) dim of transformer
N_HEADS = 8  # num of transformer heads
DROPOUT_PROB = 0.1  # dropout probability
MAX_TIMESTEP = 65536  # trajectories[0]['observations'].shape[0]
# **********************************************************************
