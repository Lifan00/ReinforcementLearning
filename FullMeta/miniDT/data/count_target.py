import pickle


def count_target(path='./perfect_v3/test_perfect_data_v2_20231112090835.pkl'):
    f = open(path, 'rb')
    trajectories = pickle.load(f)

    trajectories_len = len(trajectories)

    reward_list = []

    for i in range(trajectories_len):
        reward_list.append(sum(trajectories[i]['rewards']))

    return sum(reward_list) / len(reward_list), reward_list


if __name__ == '__main__':
    target, reward_list = count_target()
    print(reward_list)
    print(sum(reward_list))
    print(target)
