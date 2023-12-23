import copy
import os
from datetime import datetime

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import trange
# from util.data_load import getconfig
# from util.loss import getLossByWb_fn
# from util.plot_result import plotXYs_fn, plotXWbs_fn
# from util.save_read_result import saveCsvRow_fn
from util.methods import tls, ls
from now_utils import em_fn, rmse

plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号。画图之前调用
plt.rcParams['font.family'] = ['SimSun']

def test(noise_pattern):
    N_train_f = N_train = round(data.shape[0] * train_ratio)
    # 记录结果
    mid_tls_rmse = []
    mid_ls_rmse = []
    mid_em_rmse = []

    for j in trange(noise_seq_len, desc='Progress', unit='loop'):
        tmp_ls = []
        tmp_tls = []
        tmp_em = []
        copy_data = copy.deepcopy(data)
        times = j * 0.1

        for p in range(split_num):
            # 划分训练集与测试集
            np.random.seed(p)  # 保证每个训练集比例所对应的s次随机排列的顺序一致，每次排序都要使用，写在循环外不起作用
            random_datax = copy_data.reindex(np.random.permutation(copy_data.index))  # 随机排序
            X_train_random = random_datax.iloc[:N_train_f, :-1]
            Y_train_random = np.log10(np.array(random_datax.iloc[:N_train_f, -1])).reshape(-1, 1)
            X_test_random = random_datax.iloc[N_train:, :-1]
            Y_test_random = np.log10(np.array(random_datax.iloc[N_train:, -1])).reshape(-1, 1)
            standard_X = np.std(X_train_random, axis=0)
            standard_Y = np.std(Y_train_random, axis=0)

            for k in range(noise_loop):  # 生成噪声
                # 生成噪声
                np.random.seed(k)
                noise_X = np.random.randn(X_train_random.shape[0], X_train_random.shape[1])
                noise_Y = times * standard_Y * np.random.randn(Y_train_random.shape[0], 1) * noise_pattern[-1]
                # 加入噪声
                Y_train_noise = copy.deepcopy(Y_train_random) + noise_Y
                X_train_noise = copy.deepcopy(X_train_random)
                for i in range(X_train_random.shape[1]):
                    noise_X[:, i] *= times * standard_X[i] * noise_pattern[i]
                    X_train_noise.values[:, i] += noise_X[:, i]
                x_train = X_train_noise.values
                x_test = X_test_random.values

                # 最小二乘
                W_ls, b_ls, = ls(x_train, Y_train_noise)
                y_pred_ls = np.dot(x_test, W_ls) + b_ls
                tmp_ls.append(rmse(Y_test_random, y_pred_ls))
                # 总体最小二乘
                W_tls, b_tls, = tls(x_train, Y_train_noise)
                y_pred_tls = np.dot(x_test, W_tls) + b_tls
                tmp_tls.append(rmse(Y_test_random, y_pred_tls))
                # EM-TLS
                W_em, b_em, E, r = em_fn(x_train, Y_train_noise, w_epsilon, correct, max_iter_em)
                y_pred_em = np.dot(x_test, W_em) + b_em
                tmp_em.append(rmse(Y_test_random, y_pred_em))

        mid_ls_rmse.append(np.median(tmp_ls))
        mid_tls_rmse.append(np.median(tmp_tls))
        mid_em_rmse.append(np.median(tmp_em))

    print("my_ls:  ", mid_ls_rmse)
    print("my_tls: ", mid_tls_rmse)
    print("my_em:  ", mid_em_rmse)
    plt.figure(figsize=(10, 8))
    plt.plot(np.arange(0, 1.0, 0.1), mid_ls_rmse, marker='s')
    plt.plot(np.arange(0, 1.0, 0.1), mid_tls_rmse, marker='p')
    plt.plot(np.arange(0, 1.0, 0.1), mid_em_rmse, marker='*')
    plt.legend(['myLs', 'myTLS', 'em'])
    plt.xlabel('Noise Level')
    plt.ylabel('RMSE')

    title = "Noise Ratio VS RMSE\n" + '随机划分数据集次数：' + str(split_num) + ' train_ratio:' + str(train_ratio) + \
            '  随机生成噪声次数：' + str(noise_loop) + '\n噪声模式：' + str(noise_pattern) + '\n使用特征：' + str(select_feature)
    plt.title(title)
    # 进行保存和显示
    file_name = datetime.now().strftime("%Y%m%d%H%M%S") + '.png'
    if mid_em_rmse[-1] <= mid_tls_rmse[-1] and mid_em_rmse[-1] <= mid_ls_rmse[-1]:
        now_dir = os.path.join(file_dir, 'ok')
        plt.savefig(os.path.join(now_dir, file_name))
        plt.show()
    elif mid_em_rmse[-1] <= mid_tls_rmse[-1]:  # and mid_em_rmse[-1] <= mid_ls_rmse[-1]
        now_dir = os.path.join(file_dir, 'em-le-tls')
        plt.savefig(os.path.join(now_dir, file_name))
    elif mid_em_rmse[-1] <= mid_ls_rmse[-1]:
        now_dir = os.path.join(file_dir, 'em-le-ls')
        plt.savefig(os.path.join(now_dir, file_name))
    elif mid_tls_rmse[-1] <= mid_ls_rmse[-1]:
        now_dir = os.path.join(file_dir, 'tls-le-ls')
        plt.savefig(os.path.join(now_dir, file_name))
    else:
        now_dir = os.path.join(file_dir, 'other')
        plt.savefig(os.path.join(now_dir, file_name))
    print(now_dir)
    plt.show()


if __name__ == '__main__':
    select_feature = ['F2', 'F3', 'F5', 'cycle_life']  # 'F6',
    data_all = pd.read_csv('./data/dataset.csv')

    # select_feature = ['V1/D2/F2', 'Area_100_10', 'F6', 'cycle_life']  # 'F3',
    # data_all = pd.read_csv('./data/build_features.csv')

    data = data_all[select_feature]

    file_dir = 'noise_test1'
    # file_name = datetime.now().strftime("%Y%m%d%H%M%S") + '.png'
    train_ratio = 0.90
    noise_seq_len = 10
    split_num = 40
    noise_loop = 15
    w_epsilon, correct, max_iter_em = 1e-3, 1e-1, 20
    # F2 0.4~1.0 em<ls  0~0.4 tls<ls
    # F3比较小时，em<=ls
    pattern = np.array([1.0, 0.9, 0.3, 0.1])  # y的噪声越小，em tls误差越小。
    test(pattern)

    # outer_id = 0
    # for h1 in np.arange(0.0, 1.1, 0.1):
    #     for h2 in np.arange(0.0, 1.1, 0.1):
    #         for h3 in np.arange(0.0, 1.1, 0.1):
    #             # for h4 in np.arange(0.0, 1.1, 0.1):  # np.arange(0.0, 1.1, 0.2)
    #             #     for h5 in np.arange(0.0, 1.1, 0.2):
    #             # h5 = 0.1  # [0.01, 0.1, 0.5]
    #             pattern = np.array([h1, h2, h3, 0.1])
    #             print(outer_id, pattern, "============================================")
    #             test(pattern)
    #             outer_id += 1


