import copy
import os
import pandas as pd
from now_utils import *
from util.methods import ls, tls


np.set_printoptions(linewidth=np.inf)  # 设置ndaary一行显示，不折行
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号。画图之前调用
plt.rcParams['font.family'] = ['SimSun']


# 测试 em 算法
def test_em(cur_seed, plot_flag=True, save_flag=True):
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
    W_tls, b_tls = tls(X_train_random, Y_train_random)
    y_pred_tls = np.dot(X_test_random, W_tls) + b_tls
    tmp_tls = rmse(Y_test_random, y_pred_tls)[0]
    # EM-TLS
    W_em_list, b_em_list, E, r, target1, target2 = emTest_fn(X_train_random, Y_train_random, w_epsilon, correct, max_iter_em, True)
    len_em = len(W_em_list)
    # 记录em的迭代
    ls_list = [tmp_ls] * len_em
    tls_list = [tmp_tls] * len_em
    em_test_list = []
    tmp_em_wb = []
    for em_i in range(len_em):
        y_pred_em = np.dot(X_test_random, W_em_list[em_i]) + b_em_list[em_i]
        em_test_list.append(rmse(Y_test_random, y_pred_em)[0])
        tmp_em_wb.append(np.vstack((W_em_list[em_i], b_em_list[em_i])).flatten().tolist())

    # print("lapse=========================")
    # print("ls: ", tmp_ls)
    # print("tls:", tmp_tls)
    # print("em: ", em_test_list)

    fig, axs = plt.subplots(1, 3, figsize=(12, 6))
    axs[0].plot(ls_list, label='ls', marker='s')
    axs[0].plot(tls_list, label='tls', marker='p')
    axs[0].plot(em_test_list, label='em', marker='*')

    axs[1].plot(target1, label='target1', color='g', marker='*')
    axs[2].plot(target2, label='(now_x + E) @ w_std - now_y - r', color='g', marker='*')

    axs[0].legend()
    axs[1].legend()
    axs[2].legend()
    fig.suptitle('tls vs em with seed ' + str(cur_seed) + ' train_ratio:' + str(train_ratio) + '\n'+str(select_feature), fontsize=16)

    global count_em, count_tls, count_ls
    now = datetime.now()
    file_name = now.strftime("%Y%m%d%H%M%S") + str(now.microsecond // 1000).zfill(3) + '.png'
    if em_test_list[-1] <= tmp_tls:  # em_test_list[-1] <= tmp_ls and
        count_em += 1
        # print("em")
        if save_flag:
            now_dir = file_dir
            plt.savefig(os.path.join(now_dir, file_name))
    elif tmp_tls <= em_test_list[-1]:  # tmp_tls <= tmp_ls and
        count_tls += 1
        # print("tls")
    elif tmp_ls <= tmp_tls and tmp_ls <= em_test_list[-1]:
        count_ls += 1
        # print("ls")
    if plot_flag:
        plt.show()

    plt.close()
    pass


# EM中："eta:", (r_std + now_correct) / (E_std + now_correct)  最后接近1且较小结果较好。
# pearson:  ['V1/D2/F2', 'D1/F1', 'Area_100_10', 'F6', 'F3', 'F8', 'D3', 'F9', 'F7', 'F4', 'D4', 'D5/F5', 'D6']
# spearman: ['V1/D2/F2', 'Area_100_10', 'D1/F1', 'F8', 'D3', 'D6', 'F9', 'F7', 'F4', 'D4', 'F3', 'D5/F5', 'F6']
if __name__ == '__main__':
    file1, file2 = 'data/dataset.csv', 'data/build_features.csv'
    # tls/em F235：762 238 F236:474 526;  F2368:403 597 F2369:395 605;
    select_feature1 = ['F2', 'F3', 'F6', 'F9', 'cycle_life']
    # 0.8tls/em: F235：796 204  F236:429 571;   F2368:364 636  F2369:354 646;  F23569:779 221
    # 0.9tls/em: F236:395 605 F2368:349 651  F2368D3: 407 593   F23689D3:442 558
    select_feature2 = ['V1/D2/F2', 'F3', 'D5/F5', 'cycle_life']
    data_all = pd.read_csv(file2)
    select_feature = select_feature2
    data = data_all[select_feature]
    # data = data.iloc[:84]
    # print(data.iloc[:84])  # 'cell_key',

    file_dir = 'em_test'
    train_ratio = 0.9
    w_epsilon, correct, max_iter_em = 1e-6, 1e-1, 20  # correct=0.1  23579:255  0.05 251

    count_ls = 0
    count_tls = 0
    count_em = 0
    print("features:", select_feature)
    for i in range(1, 2):
        # print(f"seed {i:03d}====================================================================================")
        test_em(i, plot_flag=True, save_flag=False)
    print(count_ls, count_tls, count_em)

    pass

'''
235:eta: [0.03011115 1.0259146  1.02245453]
E_std: [3.31140775e-01 1.26665427e-05 4.65502799e-05] r_std: [0.00027214]

2358d3:
eta: [0.03022828 1.02534554 1.02232566 1.02694519 1.02600915]
E_std: [3.29749843e-01 1.61875475e-05 4.57747348e-05 5.85593804e-07 9.70925374e-06] r_std: [0.00027005]

前84条：0.8：235(1000)： 608 263 129  0.9: 499 420 81

1000(0.9): 235(1):606 160 234  235(data2):571 168 261
1000(0.8): 235(1):700 154 146  235(data2):696 152 152


'''

