import os
import copy
import numpy as np
import pandas as pd
from tqdm import trange
from datetime import datetime

from now_utils import em_fn
from util.data_load import getconfig
from util.loss import getLossByWb_fn
from util.methods import tls
from util.plot_result import plotXYs_fn, plotXWbs_fn
from util.save_read_result import saveCsvRow_fn


# 不加噪声随机划分数据集 10000 次 查看效果
def train_data_increase(train_min, train_max, step, split_num):
    seq_len = int(np.round((train_max + step - train_min) / step))
    train_sequence = [round(x, 3) for x in np.linspace(train_min, train_max + step, seq_len, endpoint=False)]
    data_size = len(data)
    test_last_10 = int(data_size * 0.1)  # 未进行四舍五入
    # print("train_sequence: ", train_sequence, len(train_sequence))

    # 0. 记录 tls 和 em 的结果
    mid_tls_train, mid_em_train = [], []
    mid_tls_rmse, mid_tls_wb = [], []
    mid_em_rmse, mid_em_wb = [], []

    start = datetime.now().strftime("%H:%M:%S")
    # 1）噪声比例以此增大
    for now_id in trange(seq_len, desc='Progress', unit='loop'):
        train_size = int(round(data_size * train_sequence[now_id]))  # todo: 不同 四舍五入了
        tmp_tls_train, tmp_em_train = [], []
        tmp_tls_rmse, tmp_tls_wb = [], []
        tmp_em_rmse, tmp_em_wb = [], []
        copy_data = copy.deepcopy(data)

        # 2）随机划分数据集
        for split in range(split_num):
            np.random.seed(split)
            random_datax = copy_data.reindex(np.random.permutation(copy_data.index))  # 随机排序
            x1_train = np.array(random_datax.iloc[:train_size, :-1])  # todo: 仅仅此处区别
            y1_train = np.log10(np.array(random_datax.iloc[:train_size, -1])).reshape(-1, 1)
            x1_test = np.array(random_datax.iloc[-test_last_10:, :-1])  # 最后的10%
            y1_test = np.log10(np.array(random_datax.iloc[-test_last_10:, -1])).reshape(-1, 1)

            # 进行训练
            tls_w1, tls_b1 = tls(x1_train, y1_train)
            tls_train_err = getLossByWb_fn(x1_train, y1_train, tls_w1, tls_b1, err_type='rmse')
            tls_test_err = getLossByWb_fn(x1_test, y1_test, tls_w1, tls_b1, err_type='rmse')
            # em 结果
            em_w2, em_b2, E, r = em_fn(x1_train, y1_train, w_epsilon, correct)
            em_train_err = getLossByWb_fn(x1_train, y1_train, em_w2, em_b2, err_type='rmse', E=E, r=r)
            em_test_err = getLossByWb_fn(x1_test, y1_test, em_w2, em_b2, err_type='rmse')

            # 4） 记录每次实验的 rmse 和 wb
            # tls
            tmp_tls_train.append(tls_train_err)
            tmp_tls_rmse.append(tls_test_err)
            tmp_tls_wb.append(np.vstack((tls_w1, tls_b1)).flatten().tolist())
            # em
            tmp_em_train.append(em_train_err)
            tmp_em_rmse.append(em_test_err)
            tmp_em_wb.append(np.vstack((em_w2, em_b2)).flatten().tolist())
            pass

        # 记录 随机划分数据集 × 随机噪声 组 的中位数
        # tls
        sort_index = np.argsort(tmp_tls_rmse)  # 从小到大，最大的放在最后
        mid_index = sort_index[len(sort_index) // 2]
        mid_err, mid_wb = tmp_tls_rmse[mid_index], tmp_tls_wb[mid_index]
        mid_tls_rmse.append(mid_err)
        mid_tls_wb.append(mid_wb)
        # tls-train
        mid_tls_train.append(np.median(tmp_tls_train))
        # em
        sort_index = np.argsort(tmp_em_rmse)
        mid_index = sort_index[len(sort_index) // 2]
        mid_err, mid_wb = tmp_em_rmse[mid_index], tmp_em_wb[mid_index]
        mid_em_rmse.append(mid_err)
        mid_em_wb.append(mid_wb)
        # em-train
        mid_em_train.append(np.median(tmp_em_train))
        pass

    end = datetime.now().strftime("%H:%M:%S")
    print(start + " -- " + end)

    if True:
        print("tls    : ", mid_tls_rmse)
        print("em     : ", mid_em_rmse)
        mid_tls_wb = np.array(mid_tls_wb)
        mid_em_wb = np.array(mid_em_wb)
        # print("中位数-tls-wb ：", mid_tls_wb)
        # print("中位数-em-wb  ：", mid_em_wb)

        seq = train_sequence
        title = "Train Size VS RMSE\n" + '随机划分数据集次数：' + str(split_num)
        x_label = 'Proportion of Training Data'
        x_err1_img, x_err2_img = 'train_train.png', 'train_test.png'
        wb_img = 'train_w.png'
        # 保存csv   类型、耗时、特征； 超参w/correct； 噪声模式； 噪声缩放比例； 训练集比例； 划分数据集； 噪声次数
        comments = ['训练集比例增大', '耗时：' + str(start) + ' -- ' + str(end), '特征选择：' + str(select_feature),
                    'hyperparameter：w_dis_epsilon：' + str(w_epsilon) + ',correct:' + str(correct),
                    'train_ratio=  ：' + str(train_min) + ' => ' + str(train_max) + '(步长' + str(step) + ')',
                    '随机划分数据集次数：' + str(split_num)]
        csv_x = 'train_size'
        csv_file = 'train.csv'

        # 1. 绘制 rmse 图像
        # plotXYs_fn(seq, [mid_tls_train, mid_em_train], x_label, 'Train RMSE',
        #            ['tls', 'em'], ['s', 'p'], NOW_DIR, x_err1_img, title)
        plotXYs_fn(seq, [mid_tls_rmse, mid_em_rmse], x_label, 'Test RMSE',
                   ['tls', 'em'], ['s', 'p'], NOW_DIR, x_err2_img, title)
        # 2. 绘制 w 和 b 随噪声变化的值。 使用前面的 x_label
        feature_len = len(select_feature)
        plotXWbs_fn(seq, [mid_tls_wb, mid_em_wb], x_label, ['tls', 'em'], ['s', 'p'], feature_len, NOW_DIR, wb_img)
        # 3. 保存训练数据
        saveCsvRow_fn(seq, [mid_tls_rmse, mid_em_rmse, mid_tls_wb.tolist(), mid_em_wb.tolist()],
                      csv_x, ['tls_rmse', 'em_rmse', 'tls_wb', 'em_wb'],
                      comments, NOW_DIR, csv_file)
    pass


# 区别只有：RES_DIR、train_data_increase
# 'V1/D2/F2', 'D1/F1', 'Area_100_10', 'F6', 'F3', 'F8', 'D3', 'F9', 'F7', 'F4', 'D4', 'D5/F5', 'D6'
if __name__ == '__main__':
    # data_path = 'data/dataset.csv'
    # select_feature = ['F2', 'F3', 'F5', 'F6' 'F9']
    data_path = 'data/build_features.csv'
    select_feature = ['V1/D2/F2', 'F3', 'F6', 'F9', 'Area_100_10']  # 'D5/F5', , 'Area_100_10'
    # select_feature = ['V1/D2/F2', 'Area_100_10', 'F6', 'F8', 'D3']  # → F2, Area, F6 F8 D3
    # data_x, data_y, convert_y = init_data(data_path, select_feature, 1)  # 全局使用

    data_all = pd.read_csv(data_path)
    data = data_all[select_feature + ['cycle_life']]
    variable = getconfig('config.json')
    w_epsilon = variable['w_epsilon']
    correct = variable['correct']
    RES_DIR = 'result_train0'  # variable["RES_DIR"]

    NOW_DIR = os.path.join(RES_DIR, datetime.now().strftime("%Y%m%d%H%M%S"))+'-2369a-4~9'
    os.makedirs(NOW_DIR)
    train_data_increase(0.4, 0.9, 0.1, split_num=10000)
    pass
