import copy
import os
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import trange
from util.data_load import getconfig
from util.loss import getLossByWb_fn
from util.methods import tls, ls
from util.plot_result import plotXYs_fn, plotXWbs_fn
from util.save_read_result import saveCsvRow_fn
from now_utils import em_fn


# xi在前面是当前循环内的变量； xi在后面是之前循环拷贝来的数据。
def noise_increase(noise_min, noise_max, step, test_ratio, split_num, noise_loop):
    seq_len = int(np.round((noise_max + step - noise_min) / step))  # 0.025 ~ 0.5 20个  0.05~0.5 10个
    noise_sequence = [round(x, 3) for x in np.linspace(noise_min, noise_max + step, seq_len, endpoint=False)]
    data_size = len(data)
    test_last_10 = int(data_size * test_ratio)  # 未进行四舍五入
    # print('noise_sequence: ', noise_sequence, len(noise_sequence))

    # 0. 记录 tls 和 em 的结果
    # mid_ls_rmse, mid_ls_wb = [], []
    # mid_tls_train, mid_em_train = [], []
    mid_tls_rmse, mid_tls_wb = [], []
    mid_em_rmse, mid_em_wb = [], []

    start = datetime.now().strftime("%H:%M:%S")
    # 1）噪声比例依次增大
    for now_id in trange(seq_len, desc='Progress', unit='loop'):
        noise_ratio = noise_sequence[now_id]
        # tmp_tls_train, tmp_em_train = [], []
        # tmp_ls_rmse, tmp_ls_wb = [], []
        tmp_tls_rmse, tmp_tls_wb = [], []
        tmp_em_rmse, tmp_em_wb = [], []
        copy_data = copy.deepcopy(data)

        # 2）随机划分数据集
        for split in range(split_num):
            np.random.seed(split)
            random_datax = copy_data.reindex(np.random.permutation(copy_data.index))  # 随机排序
            x1_train = np.array(random_datax.iloc[:-test_last_10, :-1])  # todo 仅仅此处区别
            y1_train = np.log10(np.array(random_datax.iloc[:-test_last_10, -1])).reshape(-1, 1)
            x1_test = np.array(random_datax.iloc[-test_last_10:, :-1])  # 最后的10%
            y1_test = np.log10(np.array(random_datax.iloc[-test_last_10:, -1])).reshape(-1, 1)

            x1_std_pre = np.std(x1_train, axis=0)  # 噪声标准差使用之前的 std
            y1_std_pre = np.std(y1_train, axis=0)

            # 3） 添加 q 组随机噪声。 添加噪声前 pre。
            for noise_id in range(noise_loop):
                # 1）拷贝数据
                copy_train_x2 = copy.deepcopy(x1_train)
                copy_train_y2 = copy.deepcopy(y1_train)
                copy_test_x2 = copy.deepcopy(x1_test)
                copy_test_y2 = copy.deepcopy(y1_test)
                copy_train_std_x2 = copy.deepcopy(x1_std_pre)
                copy_train_std_y2 = copy.deepcopy(y1_std_pre)

                # 2）添加噪声
                np.random.seed(noise_id)
                # a.种子不同
                x2_noise = np.random.randn(copy_train_x2.shape[0], copy_train_x2.shape[1])  # n*m
                y2_noise = np.random.randn(copy_train_y2.shape[0], 1)  # n*1
                # b.噪声比例 c.噪声模式
                x2_ratio_pattern = noise_ratio * noise_pattern[:-1]
                y2_ratio_pattern = noise_ratio * noise_pattern[-1]
                # d.x y 的标准差.   数据 → x2_with_noise， y2_with_noise
                x2_with_noise = copy.deepcopy(copy_train_x2)  # 需要拷贝，不能直接赋值？，迭代会有问题？？ n*1 * 1*1 * 1*1
                y2_with_noise = copy.deepcopy(copy_train_y2) + y2_noise * y2_ratio_pattern * copy_train_std_y2
                for i in range(copy_train_x2.shape[1]):  # 随机噪声、比例、模式、标准差
                    x2_with_noise[:, i] += x2_noise[:, i] * x2_ratio_pattern[i] * copy_train_std_x2[i]

                # 3）进行训练
                # ls_w, ls_b = ls(x2_with_noise, y2_with_noise)
                # ls_err = getLossByWb_fn(copy_test_x2, copy_test_y2, ls_w, ls_b, err_type='rmse')  # test
                tls_w1, tls_b1 = tls(x2_with_noise, y2_with_noise)
                tls_test_err = getLossByWb_fn(copy_test_x2, copy_test_y2, tls_w1, tls_b1, err_type='rmse')  # test
                em_w2, em_b2, E, r = em_fn(x2_with_noise, y2_with_noise, w_epsilon, correct, max_iter_em)
                em_test_err = getLossByWb_fn(copy_test_x2, copy_test_y2, em_w2, em_b2, err_type='rmse')
                # tls_train_err = getLossByWb_fn(copy_train_x2, copy_train_y2, tls_w1, tls_b1, err_type='rmse')  # train
                # em_train_err = getLossByWb_fn(copy_train_x2, copy_train_y2, em_w2, em_b2, err_type='rmse', E=E, r=r)

                # 4） 记录每次实验的 rmse 和 wb
                # ls
                # tmp_ls_rmse.append(ls_err)
                # tmp_ls_wb.append(np.vstack((ls_w, ls_b)).flatten().tolist())
                # tls
                # tmp_tls_train.append(tls_train_err)
                tmp_tls_rmse.append(tls_test_err)
                tmp_tls_wb.append(np.vstack((tls_w1, tls_b1)).flatten().tolist())
                # em
                # tmp_em_train.append(em_train_err)
                tmp_em_rmse.append(em_test_err)
                tmp_em_wb.append(np.vstack((em_w2, em_b2)).flatten().tolist())
                pass
            pass

        # 记录 随机划分数据集 × 随机噪声 组 的中位数
        # ls
        # sort_index = np.argsort(tmp_ls_rmse)  # 对应的数据
        # mid_index = sort_index[len(sort_index) // 2]
        # mid_err, mid_wb = tmp_ls_rmse[mid_index], tmp_ls_wb[mid_index]
        # mid_ls_rmse.append(mid_err)  # mid_...
        # mid_ls_wb.append(mid_wb)  # mid_...
        # tls
        sort_index = np.argsort(tmp_tls_rmse)  # 从小到大，最大的放在最后
        mid_index = sort_index[len(sort_index) // 2]
        mid_err, mid_wb = tmp_tls_rmse[mid_index], tmp_tls_wb[mid_index]
        mid_tls_rmse.append(mid_err)
        mid_tls_wb.append(mid_wb)
        # tls-train
        # mid_tls_train.append(np.median(tmp_tls_train))
        # em
        sort_index = np.argsort(tmp_em_rmse)
        mid_index = sort_index[len(sort_index) // 2]
        mid_err, mid_wb = tmp_em_rmse[mid_index], tmp_em_wb[mid_index]
        mid_em_rmse.append(mid_err)
        mid_em_wb.append(mid_wb)
        # em-train
        # mid_em_train.append(np.median(tmp_em_train))
        pass

    end = datetime.now().strftime("%H:%M:%S")
    print(start + " -- " + end)

    global outer_id
    if True:  # mid_ls_rmse[-1] >= mid_em_rmse[-1]:
        # 1.基本处理和判断 tls<=em 且单调递增：seq_len
        cur_dir = os.path.join(now_dir, datetime.now().strftime("%Y%m%d%H%M%S") + '-' + str(outer_id))
        ok_flag = mid_em_rmse[-1] <= mid_tls_rmse[-1]
        flag_id = 1
        while flag_id < seq_len and ok_flag:
            if mid_tls_rmse[flag_id] < mid_tls_rmse[flag_id-1] or mid_em_rmse[flag_id] < mid_em_rmse[flag_id-1]:
                ok_flag = False
            flag_id += 1
        if ok_flag:
            cur_dir = cur_dir + '√'
            print("yes")
        NOW_DIR = cur_dir
        os.makedirs(NOW_DIR)
        # print("ls     : ", mid_ls_rmse)
        print("tls    : ", mid_tls_rmse)
        print("em     : ", mid_em_rmse)
        # mid_ls_wb = np.array(mid_ls_wb)
        mid_tls_wb = np.array(mid_tls_wb)  # 转为 ndarray 否则报错：list indices must be integers or slices, not tuple
        mid_em_wb = np.array(mid_em_wb)
        # print("中位数-tls-wb ：", mid_tls_wb)
        # print("中位数-em-wb  ：", mid_em_wb)

        # 2.进行保存
        seq = noise_sequence
        title = "Noise Ratio VS RMSE\n" + '随机划分数据集次数：' + str(split_num) + '  随机生成噪声次数：' + str(
            noise_loop) + '\n' + '噪声模式：' + str(noise_pattern) + " " + str(select_feature)
        x_label = 'Increase of Noise Ratio'
        x_train_img, x_test_img = 'noise_train.png', 'noise_test.png'
        wb_img = 'noise_w.png'
        # 保存csv   类型、耗时、特征； 超参w/correct； 噪声模式； 噪声缩放比例； 训练集比例； 划分数据集； 噪声次数
        comments = ['噪声比例增大', '耗时：' + str(start) + ' -- ' + str(end), '特征选择：' + str(select_feature),
                    'hyperparameter：w_dis_epsilon：' + str(w_epsilon) + ',correct:' + str(correct),
                    'noise_pattern ：' + str(noise_pattern),
                    'noise_scale=  ：' + str(noise_min) + ' => ' + str(noise_max) + '(步长' + str(step) + ')',
                    'train_ratio   ：' + str(1 - test_ratio),
                    '随机划分数据集次数：' + str(split_num),
                    '随机生成噪声次数 ：' + str(noise_loop)]
        csv_x = 'noise_ratio'
        csv_file = 'noise.csv'

        # 1) 绘制 rmse 图像  完全一样
        plotXYs_fn(seq, [mid_tls_rmse, mid_em_rmse], x_label, 'Test RMSE',
                   ['tls', 'em'], ['p', '*'], NOW_DIR, x_test_img, title, need_show=True, need_save=True)
        # 2) 绘制 w 和 b 随噪声变化的值。 使用前面的 x_label
        feature_len = len(select_feature)
        plotXWbs_fn(seq, [mid_tls_wb, mid_em_wb], x_label, ['tls', 'em'],
                    ['p', '*'], feature_len, NOW_DIR, wb_img, need_show=False, need_save=True)
        # 3) 保存训练数据
        saveCsvRow_fn(seq, [mid_tls_rmse, mid_em_rmse, mid_tls_wb.tolist(), mid_em_wb.tolist()],
                      csv_x, ['tls_rmse', 'em_rmse', 'tls_wb', 'em_wb'],
                      comments, NOW_DIR, csv_file)

    pass


# 2369area: [27, 43, 74, 82, 91, 101, 153, 156, 164, 189, 216, 225, 228, 236, 241, 248, 273, 280, 283]  19个
# 236area：[16, 47, 48, 55, 57, 65, 91, 144, 153, 156, 166, 206, 225, 229, 247, 266, 268, 274, 296] 19个
# 区别只有：RES_DIR、noise_increase 划分数据集200次，随机噪声50次，减小噪声的影响。
# 排序后的特征： 'V1/D2/F2', 'D1/F1', 'Area_100_10', 'F6', 'F3', 'F8', 'D3', 'F9', 'F7', 'F4', 'D4', 'D5/F5', 'D6'
if __name__ == '__main__':
    file1, file2 = 'data/dataset.csv', 'data/build_features.csv'
    select_feature1 = ['F2', 'F3', 'F5', 'F6']
    # 0.8train tls/em: F235：796 204  F236:429 571;   F2368:364 636  F2369:354 646;  F23569:779 221
    select_feature2 = ['V1/D2/F2', 'F3', 'D5/F5']  # 0.9 F2368: 349 651
    data_all = pd.read_csv(file1)
    select_feature = select_feature1
    data = data_all[select_feature + ['cycle_life']]

    variable = getconfig('config.json')
    w_epsilon = variable['w_epsilon']
    correct = variable['correct']
    max_iter_em = variable['max_iter_em']
    now_dir = 'noise_test'  # 后面的字符串不同

    main_test_ratio = 0.1
    main_split_num = 60
    main_noise_loop = 60

    outer_id = -1
    noise_pattern = np.array([0.8, 0.5, 0.9, 0.5, 0.3])  # 1.0, 0.5, 0.01  [1.0, 1.0, 0.9, 0.1]5
    print(noise_pattern, "============================================")
    noise_increase(0.2, 0.9, 0.1, test_ratio=main_test_ratio, split_num=main_split_num, noise_loop=main_noise_loop)

    # # 全0的时候是由于 y 的噪声模式影响的。
    # range_list = [0.0, 0.3, 0.7, 1.0]  # np.arange(0.0, 1.1, 0.2)
    # outer_id = 0
    # for h1 in range_list:
    #     for h2 in range_list:
    #         for h3 in range_list:
    #             # for h4 in range_list:
    #             #     for h5 in range_list:
    #             noise_pattern = np.array([h1, h2, h3, 0.08])
    #             print(outer_id, noise_pattern, "============================================")
    #             noise_increase(0.1, 0.9, 0.1, test_ratio=main_test_ratio, split_num=main_split_num, noise_loop=main_noise_loop)
    #             outer_id += 1

    pass
