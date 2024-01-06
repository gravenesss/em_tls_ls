import os
import copy
import numpy as np
import pandas as pd
from tqdm import trange
from datetime import datetime
from now_utils import em_fn, lsOrTls_fn, rmse_fn
from util.plot_result import plotXYs_fn, plotXWbs_fn
from util.save_read_result import saveCsvRow_fn


# 不加噪声随机划分数据集 10000 次 查看效果
def train_data_increase(train_min, train_max, step, split_num):
    seq_len = int(np.round((train_max + step - train_min) / step))
    train_sequence = [round(x, 3) for x in np.linspace(train_min, train_max + step, seq_len, endpoint=False)]
    data_size = len(data)
    test_last_10 = int(data_size * 0.1)  # 未进行四舍五入
    all_data = copy.deepcopy(data)
    # print("train_sequence: ", train_sequence, len(train_sequence))

    # 0、记录 tls 和 em 的结果
    mid_ls_rmse, mid_ls_wb = [], []
    mid_tls_rmse, mid_tls_wb = [], []
    mid_em_rmse, mid_em_wb = [], []

    start = datetime.now().strftime("%H:%M:%S")
    # 1、噪声比例以此增大
    for now_id in trange(seq_len, desc='Progress', unit='loop'):
        train_size = int(round(data_size * train_sequence[now_id]))  # todo: 不同 四舍五入了
        tmp_ls_rmse, tmp_ls_wb = [], []
        tmp_tls_rmse, tmp_tls_wb = [], []
        tmp_em_rmse, tmp_em_wb = [], []
        copy_data = copy.deepcopy(all_data)

        # 2）随机划分数据集
        for split in range(split_num):
            np.random.seed(split)
            new_data = np.random.permutation(copy_data)  # 不改变源数据
            x1_train = new_data[:train_size, :-1]
            y1_train = np.log10(new_data[:train_size, -1]).reshape(-1, 1)
            x1_test = new_data[-test_last_10:, :-1]
            y1_test = np.log10(new_data[-test_last_10:, -1]).reshape(-1, 1)

            # 3）进行训练 并 记录每次实验的 rmse 和 wb
            # LS
            w_ls, b_ls = lsOrTls_fn(x1_train, y1_train, ls_flag=True)
            err_ls = rmse_fn(y1_test, np.dot(x1_test, w_ls) + b_ls)[0]
            tmp_ls_rmse.append(err_ls)
            tmp_ls_wb.append(np.vstack((w_ls, b_ls)).flatten().tolist())
            # TLS
            w_tls, b_tls = lsOrTls_fn(x1_train, y1_train, ls_flag=False)
            err_tls_test = rmse_fn(y1_test, np.dot(x1_test, w_tls) + b_tls)[0]
            tmp_tls_rmse.append(err_tls_test)
            tmp_tls_wb.append(np.vstack((w_tls, b_tls)).flatten().tolist())
            # EM
            w_em, b_em, E, r = em_fn(x1_train, y1_train, w_epsilon, correct, max_iter_em)
            err_em_test = rmse_fn(y1_test, np.dot(x1_test, w_em) + b_em)[0]
            tmp_em_rmse.append(err_em_test)
            tmp_em_wb.append(np.vstack((w_em, b_em)).flatten().tolist())

            pass

        # 记录 随机划分数据集 × 随机噪声 组 的中位数
        # ls
        sort_index = np.argsort(tmp_ls_rmse)  # 对应的数据
        mid_index = sort_index[len(sort_index) // 2]
        mid_err, mid_wb = tmp_ls_rmse[mid_index], tmp_ls_wb[mid_index]
        mid_ls_rmse.append(mid_err)  # mid_...
        mid_ls_wb.append(mid_wb)  # mid_...
        # tls
        sort_index = np.argsort(tmp_tls_rmse)  # 从小到大，最大的放在最后
        mid_index = sort_index[len(sort_index) // 2]
        mid_err, mid_wb = tmp_tls_rmse[mid_index], tmp_tls_wb[mid_index]
        mid_tls_rmse.append(mid_err)
        mid_tls_wb.append(mid_wb)
        # em
        sort_index = np.argsort(tmp_em_rmse)
        mid_index = sort_index[len(sort_index) // 2]
        mid_err, mid_wb = tmp_em_rmse[mid_index], tmp_em_wb[mid_index]
        mid_em_rmse.append(mid_err)
        mid_em_wb.append(mid_wb)
        pass

    end = datetime.now().strftime("%H:%M:%S")
    print(start + " -- " + end)

    if True:
        # 1.基本处理和判断 tls<=em 且单调递增：seq_len
        ok_flag = mid_em_rmse[-1] <= mid_tls_rmse[-1]
        flag_id = 1
        while flag_id < seq_len and ok_flag:
            if mid_tls_rmse[flag_id] > mid_tls_rmse[flag_id-1] or mid_em_rmse[flag_id] > mid_em_rmse[flag_id-1]:
                ok_flag = False
            flag_id += 1
        if ok_flag:
            print("yes")

        print("ls     : ", mid_ls_rmse)
        print("tls    : ", mid_tls_rmse)
        print("em     : ", mid_em_rmse)
        mid_ls_wb = np.array(mid_ls_wb)
        mid_tls_wb = np.array(mid_tls_wb)  # 转为 ndarray 否则报错：list indices must be integers or slices, not tuple
        mid_em_wb = np.array(mid_em_wb)
        # print("中位数-tls-wb ：", mid_tls_wb)
        # print("中位数-em-wb  ：", mid_em_wb)

        seq = train_sequence
        title = "Train Size VS RMSE\n" + '随机划分数据集次数：' + str(split_num) +\
                '\nfeature:' + str(select_feature) + '  w_dis_epsilon：' + str(w_epsilon) + ', correct: ' + str(correct)
        x_label = 'Proportion of Training Data'
        x_train_img, x_test_img = 'train_train.png', 'train_test.png'
        wb_img = 'train_w.png'
        # 保存csv   类型、耗时、特征； 超参w/correct； 噪声模式； 噪声缩放比例； 训练集比例； 划分数据集； 噪声次数
        comments = ['训练集比例增大', '耗时：' + str(start) + ' -- ' + str(end), '特征选择：' + str(select_feature),
                    'hyperparameter：w_dis_epsilon：' + str(w_epsilon) + ',correct:' + str(correct),
                    'train_ratio=  ：' + str(train_min) + ' => ' + str(train_max) + '(步长' + str(step) + ')',
                    '随机划分数据集次数：' + str(split_num)]
        csv_x = 'train_size'
        csv_file = 'train.csv'

        # 1) 绘制 rmse 图像  完全一样
        plotXYs_fn(seq, [mid_tls_rmse, mid_em_rmse], x_label, 'Test RMSE',
                   ['tls', 'em'], ['p', '*'], NOW_DIR, x_test_img, title, need_show=True, need_save=True)
        # plotXYs_fn(seq, [mid_ls_rmse, mid_tls_rmse, mid_em_rmse], x_label, 'Test RMSE',
        #            ['ls', 'tls', 'em'], ['s', 'p', '*'], NOW_DIR, x_test_img, title)
        # 2) 绘制 w 和 b 随噪声变化的值。 使用前面的 x_label
        feature_len = len(select_feature)
        plotXWbs_fn(seq, [mid_tls_wb, mid_em_wb], x_label, ['tls', 'em'],
                    ['p', '*'], feature_len, NOW_DIR, wb_img, need_show=False, need_save=True)
        # 3) 保存训练数据
        saveCsvRow_fn(seq, [mid_tls_rmse, mid_em_rmse, mid_tls_wb.tolist(), mid_em_wb.tolist()],
                      csv_x, ['tls_rmse', 'em_rmse', 'tls_wb', 'em_wb'],
                      comments, NOW_DIR, csv_file)
    pass


# 区别只有：RES_DIR、train_data_increase
# 'V1/D2/F2', 'D1/F1', 'Area_100_10', 'F6', 'F3', 'F8', 'D3', 'F9', 'F7', 'F4', 'D4', 'D5/F5', 'D6'
if __name__ == '__main__':
    file1, file2 = 'data/dataset.csv', 'data/build_features.csv'
    select_feature1 = ['F2', 'F3', 'F5', 'F6']
    select_feature2 = ['V1/D2/F2', 'F6', 'F7']  # 20240102: 236(595/1000)， 2368(642/1000)

    # 选择文件和特征
    data_all = pd.read_csv(file2)
    select_feature = select_feature2
    data = (data_all[select_feature + ['cycle_life']]).values

    # 初始化参数
    max_iter_em, w_epsilon, correct = 20, 1e-6, 0.1

    # 文件夹
    now_dir = 'test_train0'
    NOW_DIR = os.path.join(now_dir, '267D3-'+datetime.now().strftime("%Y%m%d%H%M%S"))
    os.makedirs(NOW_DIR)
    train_data_increase(0.1, 0.9, 0.1, split_num=10000)
    pass
