import copy

import pandas as pd
from sklearn.model_selection import train_test_split
from now_utils import *
from util.data_load import init_data, getconfig


# 测试 em 算法
def test_em(cur_seed, plot_flag=True):
    N_train_f = N_train = round(data.shape[0] * train_ratio)
    copy_data = copy.deepcopy(data)
    np.random.seed(cur_seed)  # 保证每个训练集比例所对应的s次随机排列的顺序一致，每次排序都要使用，写在循环外不起作用
    random_datax = copy_data.reindex(np.random.permutation(copy_data.index))  # 随机排序
    X_train_random = random_datax.iloc[:N_train_f, :-1]
    Y_train_random = np.log10(np.array(random_datax.iloc[:N_train_f, -1])).reshape(-1, 1)
    X_test_random = random_datax.iloc[N_train:, :-1]
    Y_test_random = np.log10(np.array(random_datax.iloc[N_train:, -1])).reshape(-1, 1)


    train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=test_ratio, random_state=cur_seed)
    # em 结果 em中就可以确定是否还原y 计算rmse
    em_err, tls_err = emTest_fn(train_x, train_y, test_x, test_y, cur_seed, w_epsilon, correct, plot_pic=plot_flag)

    global count
    if em_err < tls_err:
        count += 1
        print("yes")

    pass


# 'V1/D2/F2', 'D1/F1', 'Area_100_10', 'F6', 'F3', 'F8', 'D3', 'F9', 'F7', 'F4', 'D4', 'D5/F5', 'D6'
# select_feature = ['V1/D2/F2', 'Area_100_10', 'F6', 'F8', 'D3']  # → F2, Area, F6 F8 D3
if __name__ == '__main__':
    select_feature = ['F2', 'F3', 'F5', 'cycle_life']  # 'F6',
    data_all = pd.read_csv('./data/dataset.csv')
    # select_feature = ['V1/D2/F2', 'Area_100_10', 'F6', 'cycle_life']  # 'F3',
    # data_all = pd.read_csv('./data/build_features.csv')

    data = data_all[select_feature]
    file_dir = 'em_test'
    # file_name = datetime.now().strftime("%Y%m%d%H%M%S") + '.png'
    train_ratio = 0.90
    noise_seq_len = 10
    split_num = 40
    noise_loop = 15
    w_epsilon, correct, max_iter_em = 1e-3, 1e-1, 20

    count = 0
    for i in range(0, 10):
        print(f"seed {i:02d}=================================")
        test_em(0.1, i, True)
    print(count)

    pass
