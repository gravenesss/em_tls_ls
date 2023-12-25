import copy
import os
from datetime import datetime
import pandas as pd
from now_utils import *
from util.methods import ls, tls


np.set_printoptions(linewidth=np.inf)  # 设置ndaary一行显示，不折行
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号。画图之前调用
plt.rcParams['font.family'] = ['SimSun']


# 测试 em 算法
def test_em(cur_seed, plot_flag=True):
    N_train_f = N_train = round(data.shape[0] * train_ratio)
    copy_data = copy.deepcopy(data)
    np.random.seed(cur_seed)  # 保证每个训练集比例所对应的s次随机排列的顺序一致，每次排序都要使用，写在循环外不起作用
    random_datax = copy_data.reindex(np.random.permutation(copy_data.index))  # 随机排序
    X_train_random = random_datax.iloc[:N_train_f, :-1].values
    Y_train_random = np.log10(np.array(random_datax.iloc[:N_train_f, -1])).reshape(-1, 1)
    X_test_random = random_datax.iloc[N_train:, :-1].values
    Y_test_random = np.log10(np.array(random_datax.iloc[N_train:, -1])).reshape(-1, 1)

    W_ls, b_ls, = ls(X_train_random, Y_train_random)
    y_pred_ls = np.dot(X_test_random, W_ls) + b_ls
    tmp_ls = rmse(Y_test_random, y_pred_ls)[0]
    # 总体最小二乘
    W_tls, b_tls, = tls(X_train_random, Y_train_random)
    y_pred_tls = np.dot(X_test_random, W_tls) + b_tls
    tmp_tls = rmse(Y_test_random, y_pred_tls)[0]
    # EM-TLS
    W_em_list, b_em_list, E, r, target1, target2 = emTest_fn(X_train_random, Y_train_random, w_epsilon, correct, max_iter_em)
    len_em = len(W_em_list)
    # 记录em的迭代
    tmp_ls_list = [tmp_ls] * len_em
    tmp_tls_list = [tmp_tls] * len_em
    em_test_list = []
    for em_i in range(len_em):
        y_pred_em = np.dot(X_test_random, W_em_list[em_i]) + b_em_list[em_i]
        em_test_list.append(rmse(Y_test_random, y_pred_em)[0])

    # print("lapse=========================")
    # print("len_em", len_em, "target_len", len(target_list))
    # print("ls: ", tmp_ls)
    # print("tls:", tmp_tls)
    # print("em: ", em_test_list)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].plot(tmp_ls_list, label='ls', marker='s')
    axs[0].plot(tmp_tls_list, label='tls', marker='p')
    axs[0].plot(em_test_list, label='em', marker='*')
    axs[0].legend()
    axs[1].plot(target1, label='target1', color='g', marker='*')
    # axs[1].plot(target2, label='target2', marker='*')
    axs[1].legend()
    fig.suptitle('tls vs em with seed ' + str(cur_seed) + ' train_ratio:' + str(train_ratio) + '\n'+str(select_feature), fontsize=16)

    global count
    if em_test_list[-1] <= tmp_tls <= tmp_ls:
        count += 1
        print("yes")
        if plot_flag:
            now_dir = file_dir
            plt.savefig(os.path.join(now_dir, file_name))

    if plot_flag:
        plt.show()

    pass


# pearson:  ['V1/D2/F2', 'D1/F1', 'Area_100_10', 'F6', 'F3', 'F8', 'D3', 'F9', 'F7', 'F4', 'D4', 'D5/F5', 'D6']
# spearman: ['V1/D2/F2', 'Area_100_10', 'D1/F1', 'F8', 'D3', 'D6', 'F9', 'F7', 'F4', 'D4', 'F3', 'D5/F5', 'F6']
if __name__ == '__main__':
    # 23569:166    2356:141     # 23579:247    2357:225  235:233  236:21   #  235789:212  235679:148  23567:130
    # 5000: 23579:1187  235:1181
    select_feature = ['F2', 'F3', 'F5', 'F6', 'cycle_life']
    data_all = pd.read_csv('./data/dataset.csv')

    # select_feature = ['Area_100_10', 'V1/D2/F2', 'F5', 'cycle_life']  # Area23: 66  Area36:68
    # data_all = pd.read_csv('./data/build_features.csv')

    data = data_all[select_feature]
    file_dir = 'em_test'
    train_ratio = 0.9
    w_epsilon, correct, max_iter_em = 1e-6, 0.1, 20  # correct=0.1  23579:255  0.05 251

    count = 0
    for i in range(0, 5000):
        print(f"seed {i:03d}====================================================================================")
        now = datetime.now()
        file_name = now.strftime("%Y%m%d%H%M%S") + str(now.microsecond // 1000).zfill(3) + '.png'
        test_em(i, False)
    print(count)

    pass
