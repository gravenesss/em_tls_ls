import copy
from datetime import datetime
from tqdm import trange
from now_utils import em_fn, modelPredict_fn
from util.data_load import init_data, getconfig
from util.feature_select import *
from util.loss import getLossByWb_fn
from util.methods import tls
from util.plot_result import *
from util.save_read_result import *


def train_data_increase(train_min, train_max, step, noise_ratio, split_num, noise_loop):
    seq_len = int(np.round((train_max + step - train_min) / step))
    train_sequence = [round(x, 3) for x in np.linspace(train_min, train_max + step, seq_len, endpoint=False)]
    data_size = len(data_x)
    test_last_10 = int(data_size * 0.1)  # 未进行四舍五入
    print(train_sequence, len(train_sequence))

    # 0. 记录 tls 和 em 的结果
    mid_tls_train, mid_em_train = [], []
    mid_tls_rmse, mid_tls_wb = [], []
    mid_em_rmse, mid_em_wb = [], []
    mid_linear_rmse = []

    start = datetime.now().strftime("%H:%M:%S")
    # 1）噪声比例以此增大
    for now_id in trange(seq_len, desc='Progress', unit='loop'):
        train_size = int(round(data_size * train_sequence[now_id]))  # todo: 不同 四舍五入了
        tmp_tls_train, tmp_em_train = [], []
        tmp_tls_rmse, tmp_tls_wb = [], []
        tmp_em_rmse, tmp_em_wb = [], []
        tmp_linear_rmse = []
        copy_x0 = copy.deepcopy(data_x)
        copy_y0 = copy.deepcopy(data_y)

        # 2）随机划分数据集
        for split in range(split_num):
            # 拷贝数据
            copy_all_x1 = copy.deepcopy(copy_x0)
            copy_all_y1 = copy.deepcopy(copy_y0)

            # 随机排序进行划分
            np.random.seed(split)
            random_indices = np.random.permutation(data_size)  # 随机排序
            x1_data = copy_all_x1[random_indices]
            y1_data = copy_all_y1[random_indices]
            # 都只 选取最后 test_ratio 的数据作为 测试集合  数据。。。下4行
            x1_test = x1_data[-test_last_10:]
            y1_test = y1_data[-test_last_10:]
            x1_train = x1_data[:train_size]  # todo 仅仅此处区别
            y1_train = y1_data[:train_size]
            x1_std_pre = np.std(x1_train, axis=0)  # 噪声标准差使用之前的 std
            y1_std_pre = np.std(y1_train, axis=0)

            # 3） 添加 q 组随机噪声。 添加噪声前 pre、添加噪声后 now、标准化后 new。
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
                x2_noise_ratio_pattern = noise_ratio * noise_pattern[:-1]
                y2_noise_ratio_pattern = noise_ratio * noise_pattern[-1]
                # d.x y 的标准差.   数据。。。
                x2_with_noise = copy.deepcopy(copy_train_x2)  # 需要拷贝，不能直接赋值？，迭代会有问题？？ n*1 * 1*1 * 1*1
                y2_with_noise = copy.deepcopy(copy_train_y2) + y2_noise * y2_noise_ratio_pattern * copy_train_std_y2
                for i in range(copy_train_x2.shape[1]):  # 随机噪声、比例、模式、标准差
                    x2_with_noise[:, i] += x2_noise[:, i] * x2_noise_ratio_pattern[i] * copy_train_std_x2[i]

                # 进行训练
                tls_w1, tls_b1 = tls(x2_with_noise, y2_with_noise)
                tls_train_err = getLossByWb_fn(copy_train_x2, copy_train_y2, tls_w1, tls_b1, err_type='rmse')
                tls_test_err = getLossByWb_fn(copy_test_x2, copy_test_y2, tls_w1, tls_b1, err_type='rmse')
                # em 结果
                em_w2, em_b2, E, r = em_fn(x2_with_noise, y2_with_noise, copy_test_x2, copy_test_y2, w_epsilon, correct)
                em_train_err = getLossByWb_fn(copy_train_x2, copy_train_y2, em_w2, em_b2, err_type='rmse', E=E, r=r)
                em_test_err = getLossByWb_fn(copy_test_x2, copy_test_y2, em_w2, em_b2, err_type='rmse')

                # 4） 记录每次实验的 rmse 和 wb
                # tls
                tmp_tls_train.append(tls_train_err)
                tmp_tls_rmse.append(tls_test_err)
                tmp_tls_wb.append(np.vstack((tls_w1, tls_b1)).flatten().tolist())
                # em
                tmp_em_train.append(em_train_err)
                tmp_em_rmse.append(em_test_err)
                tmp_em_wb.append(np.vstack((em_w2, em_b2)).flatten().tolist())
                # linear
                tmp_linear_rmse.append(modelPredict_fn(x2_with_noise, y2_with_noise, copy_test_x2, copy_test_y2, 'linear'))
                pass
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

        # linear
        tmp_linear_rmse.sort(reverse=False)
        mid_linear_rmse.append(tmp_linear_rmse[len(tmp_linear_rmse) // 2])
        pass

    end = datetime.now().strftime("%H:%M:%S")
    print(start + " -- " + end)

    if True:
        print("tls    : ", mid_tls_rmse)
        print("em     : ", mid_em_rmse)
        print("linear : ", mid_linear_rmse)
        mid_tls_wb = np.array(mid_tls_wb)
        mid_em_wb = np.array(mid_em_wb)
        # print("中位数-tls-wb ：", mid_tls_wb)
        # print("中位数-em-wb  ：", mid_em_wb)

        seq = train_sequence
        title = "Train Size VS RMSE\n" + '随机划分数据集次数：' + str(split_num) + '  随机生成噪声次数：' + str(noise_loop)
        x_label = 'Proportion of Training Data'
        x_err1_img, x_err2_img = 'train_train.png', 'train_test.png'
        wb_img = 'train_w.png'
        # 保存csv   类型、耗时、特征； 超参w/correct； 噪声模式； 噪声缩放比例； 训练集比例； 划分数据集； 噪声次数
        comments = ['训练集比例增大', '耗时：' + str(start) + ' -- ' + str(end), '特征选择：' + str(select_feature),
                    'hyperparameter：w_dis_epsilon：' + str(w_epsilon) + ',correct:' + str(correct),
                    'noise_pattern ：' + str(noise_pattern),
                    'noise_scale   ：' + str(noise_ratio),
                    'train_ratio=  ：' + str(train_min) + ' => ' + str(train_max) + '(步长' + str(step) + ')',
                    '随机划分数据集次数：' + str(split_num),
                    '随机生成噪声次数 ：' + str(noise_loop)]
        csv_x = 'train_size'
        csv_file = 'train.csv'

        # 1. 绘制 rmse 图像
        # plotXYs_fn(seq, [mid_tls_rmse, mid_em_rmse, mid_linear_rmse], x_label, 'Test RMSE',
        #            ['tls', 'em', 'linear'], ['s', 'p', 'o'], NOW_DIR, x_err1_img, title)
        plotXYs_fn(seq, [mid_tls_rmse, mid_em_rmse], x_label, 'Test RMSE',
                   ['tls', 'em'], ['s', 'p'], NOW_DIR, x_err1_img, title)
        plotXYs_fn(seq, [mid_tls_train, mid_em_train], x_label, 'Train RMSE',
                   ['tls', 'em'], ['s', 'p'], NOW_DIR, x_err2_img, title)
        # 2. 绘制 w 和 b 随噪声变化的值。 使用前面的 x_label
        feature_len = len(select_feature)
        plotXWbs_fn(seq, [mid_tls_wb, mid_em_wb], x_label, ['tls', 'em'], ['s', 'p'], feature_len, NOW_DIR, wb_img)
        # 3. 保存训练数据
        saveCsvRow_fn(seq, [mid_tls_rmse, mid_em_rmse, mid_linear_rmse, mid_tls_wb.tolist(), mid_em_wb.tolist()],
                      csv_x, ['tls_rmse', 'em_rmse', 'linear_rmse', 'tls_wb', 'em_wb'],
                      comments, NOW_DIR, csv_file)
    pass


# 区别只有：RES_DIR、train_data_increase
if __name__ == '__main__':
    # data_path = 'data/dataset.csv'
    # select_feature = [9, 10, 12, 13, 16]  # 2 3 5 6 9
    data_path = 'data/build_features.csv'
    # 'V1/D2/F2', 'D1/F1', 'Area_100_10', 'F6', 'F3', 'F8', 'D3', 'F9', 'F7', 'F4', 'D4', 'D5/F5', 'D6'
    select_feature = ['V1/D2/F2', 'F3', 'F6', 'F9', 'Area_100_10']  # 'D5/F5',
    # select_feature = ['V1/D2/F2', 'Area_100_10', 'F6', 'F8', 'D3']  # → F2, Area, F6 F8 D3
    data_x, data_y, convert_y = init_data(data_path, select_feature, 1)  # 全局使用
    variable = getconfig('config.json')
    w_epsilon = variable['w_epsilon']
    correct = variable['correct']
    RES_DIR = 'result_train'  # variable["RES_DIR"]

    only_test = False
    random_seeds = list(range(101, 102))
    for random_id in random_seeds:  # trange(len(random_seeds), desc='Random Process', unit='loop'):
        np.random.seed(random_id)  # random_seeds[random_id]
        noise_pattern = np.random.uniform(0.2, 2, 6)
        # noise_pattern = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        # noise_pattern = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        print(random_id, noise_pattern, "============================================")

        NOW_DIR = os.path.join(RES_DIR, datetime.now().strftime("%Y%m%d%H%M%S") + '-' + str(random_id))
        os.makedirs(NOW_DIR)
        if only_test:
            train_data_increase(0.2, 0.2, 0.1, noise_ratio=0.2, split_num=100, noise_loop=100)
        else:
            train_data_increase(0.2, 0.9, 0.1, noise_ratio=0.2, split_num=100, noise_loop=100)
    pass
