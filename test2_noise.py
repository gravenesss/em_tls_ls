import copy
import os
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import trange
from util.loss import getLossByWb_fn
from util.plot_result import plotXYs_fn, plotXWbs_fn
from util.save_read_result import saveCsvRow_fn
from now_utils import em_fn, tls, ls, rmse


# xi在前面是当前循环内的变量； xi在后面是之前循环拷贝来的数据。
def noise_increase(noise_min, noise_max, step, train_ratio, split_num, noise_loop):
    seq_len = int(np.round((noise_max + step - noise_min) / step))  # 0.05~0.5(0.05) 10个
    noise_sequence = [round(x, 3) for x in np.linspace(noise_min, noise_max + step, seq_len, endpoint=False)]
    train_num = round(data.shape[0] * train_ratio)  # 进行四舍五入
    all_data = copy.deepcopy(data)
    # print('noise_sequence: ', noise_sequence, len(noise_sequence))

    # 0. 记录 tls 和 em 的结果
    mid_ls_rmse, mid_ls_wb = [], []
    mid_tls_rmse, mid_tls_wb = [], []
    mid_em_rmse, mid_em_wb = [], []

    start = datetime.now().strftime("%H:%M:%S")
    # 1、噪声比例依次增大
    for now_id in trange(seq_len, desc='Progress', unit='loop'):
        noise_ratio = noise_sequence[now_id]
        tmp_ls_rmse, tmp_ls_wb = [], []
        tmp_tls_rmse, tmp_tls_wb = [], []
        tmp_em_rmse, tmp_em_wb = [], []
        copy_data = copy.deepcopy(all_data)

        # 2、随机划分数据集
        for split in range(split_num):
            np.random.seed(split)
            new_data = np.random.permutation(copy_data)  # 不改变源数据
            x1_train = new_data[:train_num, :-1]  # todo 对于test_train仅仅此处区别
            y1_train = np.log10(new_data[:train_num, -1]).reshape(-1, 1)
            x1_test = new_data[train_num:, :-1]
            y1_test = np.log10(new_data[train_num:, -1]).reshape(-1, 1)
            # 计算标准差：噪声标准差使用之前的 std
            x1_std_pre = np.std(x1_train, axis=0)
            y1_std_pre = np.std(y1_train, axis=0)

            # 3、添加 q 组随机噪声。 添加噪声前 pre。
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
                # 噪声模式和噪声比例
                x2_ratio_pattern = noise_ratio * noise_pattern[:-1]
                y2_ratio_pattern = noise_ratio * noise_pattern[-1]
                # 生成噪声：+ std
                x2_noise = np.zeros((copy_train_x2.shape[0], copy_train_x2.shape[1]))
                for i in range(copy_train_x2.shape[1]):  # .ravel()因为前面是(112,) 后面是(112,1) 报错：could not broadcast
                    x2_noise[:, i] = np.random.normal(loc=0, scale=x2_ratio_pattern[i], size=(copy_train_x2.shape[0], 1)).ravel() * copy_train_std_x2[i]
                y2_noise = np.random.normal(loc=0, scale=y2_ratio_pattern, size=(copy_train_y2.shape[0], 1)) * copy_train_std_y2
                # 数据加入噪声
                x2_with_noise = copy.deepcopy(copy_train_x2) + x2_noise
                y2_with_noise = copy.deepcopy(copy_train_y2) + y2_noise

                # 3）进行训练 并 记录每次实验的 rmse 和 wb
                # LS
                w_ls, b_ls = ls(x2_with_noise, y2_with_noise)
                err_ls = rmse(copy_test_y2, np.dot(copy_test_x2, w_ls) + b_ls)[0]
                tmp_ls_rmse.append(err_ls)
                tmp_ls_wb.append(np.vstack((w_ls, b_ls)).flatten().tolist())
                # TLS
                w_tls, b_tls = tls(x2_with_noise, y2_with_noise)
                err_tls_test = rmse(copy_test_y2, np.dot(copy_test_x2, w_tls) + b_tls)[0]
                tmp_tls_rmse.append(err_tls_test)
                tmp_tls_wb.append(np.vstack((w_tls, b_tls)).flatten().tolist())
                # EM
                w_em, b_em = em_fn(x2_with_noise, y2_with_noise, w_epsilon, correct, max_iter_em)
                err_em_test = rmse(copy_test_y2, np.dot(copy_test_x2, w_em) + b_em)[0]
                tmp_em_rmse.append(err_em_test)
                tmp_em_wb.append(np.vstack((w_em, b_em)).flatten().tolist())

                pass

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

    if True:  # mid_ls_rmse[-1] >= mid_em_rmse[-1]:
        # 1.基本处理和判断 tls<=em 且单调递增：seq_len
        cur_dir = os.path.join(now_dir, datetime.now().strftime("%Y%m%d%H%M%S"))
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
        print("ls     : ", mid_ls_rmse)
        print("tls    : ", mid_tls_rmse)
        print("em     : ", mid_em_rmse)
        mid_ls_wb = np.array(mid_ls_wb)
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
                    'train_ratio   ：' + str(train_ratio),
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
    select_feature2 = ['V1/D2/F2', 'F6', 'F3', 'F8']  # 20240102: 236(595/1000)， 2368(642/1000)

    # 选择文件和特征
    data_all = pd.read_csv(file2)
    select_feature = select_feature2
    data = (data_all[select_feature + ['cycle_life']]).values

    # 初始化参数
    main_train_ratio = 0.9
    max_iter_em, w_epsilon, correct = 20, 1e-6, 1e-2
    # 划分+噪声模式
    main_split_num = 100
    main_noise_loop = 60
    noise_pattern = np.array([0.1, 0.2, 0.3, 0.4, 0.1])

    # print(noise_pattern, "============================================")
    now_dir = 'test_noise'  # 后面的字符串不同
    noise_increase(0.0, 0.9, 0.1, train_ratio=main_train_ratio, split_num=main_split_num, noise_loop=main_noise_loop)

    pass
