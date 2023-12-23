import copy
from datetime import datetime
from tqdm import trange
from util.data_load import init_data, getconfig
from util.feature_select import *
from util.loss import getLossByWb_fn
from util.ls_tls import ls_fn
from util.plot_result import *
from util.save_read_result import *
from now_utils import *


NOW_DIR = ''
noise_pattern = np.array([])


# xi在前面是当前循环内的变量； xi在后面是之前循环拷贝来的数据。
def noise_increase(noise_min, noise_max, step, test_ratio, split_num, noise_loop):
    seq_len = int(np.round((noise_max + step - noise_min) / step))  # 0.025 ~ 0.5 20个  0.05~0.5 10个
    noise_sequence = [round(x, 3) for x in np.linspace(noise_min, noise_max + step, seq_len, endpoint=False)]
    data_size = len(data_x)
    test_last_10 = int(data_size * test_ratio)  # 未进行四舍五入
    print('noise_sequence: ', noise_sequence, len(noise_sequence))

    # 0. 记录 tls 和 em 的结果
    mid_ls_rmse, mid_ls_wb = [], []
    mid_tls_rmse, mid_tls_wb = [], []
    mid_em_rmse, mid_em_wb = [], []
    mid_linear_rmse = []
    mid_elastic_rmse = []
    mid_lasso_rmse = []

    start = datetime.now().strftime("%H:%M:%S")
    # 1）噪声比例依次增大      # for now_id, noise_ratio in enumerate(noise_sequence):
    for now_id in trange(seq_len, desc='Progress', unit='loop'):
        noise_ratio = noise_sequence[now_id]
        tmp_ls_rmse, tmp_ls_wb = [], []
        tmp_tls_rmse, tmp_tls_wb = [], []
        tmp_em_rmse, tmp_em_wb = [], []
        tmp_linear_rmse = []
        tmp_elastic_rmse = []
        tmp_lasso_rmse = []
        copy_x0 = copy.deepcopy(data_x)
        copy_y0 = copy.deepcopy(data_y)

        # 2）随机划分数据集
        for split in range(split_num):
            # 拷贝数据
            copy_all_x1 = copy.deepcopy(copy_x0)
            copy_all_y1 = copy.deepcopy(copy_y0)
            # 随机排序进行划分
            np.random.seed(split)
            random_indices = np.random.permutation(data_size)  # 把所有的数据集 随机排序。数据。。。
            data_x1 = copy_all_x1[random_indices]
            data_y1 = copy_all_y1[random_indices]
            # 都只 选取最后 test_ratio 的数据作为 测试集合  数据。。。下4行
            x1_test = data_x1[-test_last_10:]
            y1_test = data_y1[-test_last_10:]
            x1_train = data_x1[:-test_last_10]
            y1_train = data_y1[:-test_last_10]
            m = x1_train.shape[1]
            x1_std_pre = np.std(x1_train, axis=0)  # 噪声标准差使用之前的 std
            y1_std_pre = np.std(y1_train, axis=0)

            # 3） 添加 q 组随机噪声。 添加噪声前 pre、添加噪声后 now、标准化后 new。
            for noise_id in range(noise_loop):
                # 0）拷贝数据
                copy_train_x2 = copy.deepcopy(x1_train)
                copy_train_y2 = copy.deepcopy(y1_train)
                copy_train_std_x2 = copy.deepcopy(x1_std_pre)
                copy_train_std_y2 = copy.deepcopy(y1_std_pre)
                copy_test_x2 = copy.deepcopy(x1_test)
                copy_test_y2 = copy.deepcopy(y1_test)

                # 1）添加噪声
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

                # 2）进行标准化
                x2_with_noise_mean = np.mean(x2_with_noise, axis=0).reshape(-1, 1)
                y2_with_noise_mean = np.mean(y2_with_noise, axis=0).reshape(-1, 1)
                x2_with_noise_std = np.std(x2_with_noise, axis=0).reshape(-1, 1)  # em中使用添加噪声之后的 std
                y2_with_noise_std = np.std(y2_with_noise, axis=0).reshape(-1, 1)
                x2_after_std = (x2_with_noise - x2_with_noise_mean.T) / x2_with_noise_std.T  # (112,5)*(5,1)转置使用广播机制
                y2_after_std = (y2_with_noise - y2_with_noise_mean) / y2_with_noise_std

                # 3) 进行训练 并还原计算rmse。 下面的方法中内部都没有进行标准化的：ls_fn、tls_fn、em_fn
                # ls
                # ls_w0_std = ls_fn(x2_after_std, y2_after_std)
                # ls_w0, ls_b0 = getWb_fn(m, ls_w0_std, x2_with_noise_std, x2_with_noise_mean, y2_with_noise_std, y2_with_noise_mean)
                # ls_err = getLossByWb_fn(copy_test_x2, copy_test_y2, ls_w0, ls_b0, err_type='rmse')
                # ls
                ls_w0_std = ls_fn(x2_after_std, y2_after_std)
                ls_w0, ls_b0 = getWb_fn(m, ls_w0_std, x2_with_noise_std, x2_with_noise_mean, y2_with_noise_std, y2_with_noise_mean)
                ls_err = getLossByWb_fn(copy_test_x2, copy_test_y2, ls_w0, ls_b0, err_type='rmse')
                # tls
                tls_w1_std = tls_fn(x2_after_std, y2_after_std)  # 使用已经进行标准化的数据。tls内部未进行标准化  w未进行还原
                tls_w1, tls_b1 = getWb_fn(m, tls_w1_std, x2_with_noise_std, x2_with_noise_mean, y2_with_noise_std, y2_with_noise_mean)
                tls_err = getLossByWb_fn(copy_test_x2, copy_test_y2, tls_w1, tls_b1, err_type='rmse')
                # em 结果
                em_err, em_wb2 = em_fn(x2_with_noise, y2_with_noise, copy_test_x2, copy_test_y2, w_epsilon, correct)

                # 4） 记录每次实验的 rmse 和 wb
                # ls
                tmp_ls_rmse.append(ls_err)
                tmp_ls_wb.append(np.vstack((ls_w0, ls_b0)).flatten().tolist())
                # tls
                tmp_tls_rmse.append(tls_err)
                tmp_tls_wb.append(np.vstack((tls_w1, tls_b1)).flatten().tolist())
                # em
                tmp_em_rmse.append(em_err)
                tmp_em_wb.append(em_wb2.flatten().tolist())
                # linear 添加噪声后的数据: convert_y 默认为'1'
                tmp_linear_rmse.append(modelPredict_fn(x2_after_std, y2_after_std, copy_test_x2, copy_test_y2, 'linear'))
                # elasticNet
                tmp_elastic_rmse.append(modelPredict_fn(x2_after_std, y2_after_std, copy_test_x2, copy_test_y2, 'en'))
                # Lasso
                tmp_lasso_rmse.append(modelPredict_fn(x2_after_std, y2_after_std, copy_test_x2, copy_test_y2, 'lasso'))

        # 记录 随机划分数据集 × 随机噪声 组 的中位数
        # ls
        sorted_data = sorted(zip(tmp_ls_rmse, tmp_ls_wb))  # 对应的数据
        mid_index = len(sorted_data) // 2
        mid_err, mid_wb = sorted_data[mid_index]
        mid_ls_rmse.append(mid_err)  # mid_...
        mid_ls_wb.append(mid_wb)  # mid_...
        # tls
        sorted_data = sorted(zip(tmp_tls_rmse, tmp_tls_wb))
        mid_index = len(sorted_data) // 2
        mid_err, mid_wb = sorted_data[mid_index]
        mid_tls_rmse.append(mid_err)
        mid_tls_wb.append(mid_wb)
        # em
        sorted_data = sorted(zip(tmp_em_rmse, tmp_em_wb))
        mid_index = len(sorted_data) // 2
        mid_err, mid_wb = sorted_data[mid_index]
        mid_em_rmse.append(mid_err)
        mid_em_wb.append(mid_wb)
        # linear
        tmp_linear_rmse.sort(reverse=False)
        mid_linear_rmse.append(tmp_linear_rmse[len(tmp_linear_rmse) // 2])
        # elasticNet
        tmp_elastic_rmse.sort(reverse=False)
        mid_elastic_rmse.append(tmp_elastic_rmse[len(tmp_elastic_rmse) // 2])
        # Lasso
        tmp_lasso_rmse.sort(reverse=False)
        mid_lasso_rmse.append(tmp_lasso_rmse[len(tmp_lasso_rmse) // 2])

    end = datetime.now().strftime("%H:%M:%S")
    print(start + " -- " + end)
    print("ls     : ", mid_ls_rmse)
    print("tls    : ", mid_tls_rmse)
    print('em     : ', mid_em_rmse)
    # plt.plot(noise_sequence, mid_ls_rmse)
    # plt.plot(noise_sequence, mid_tls_rmse)
    # plt.plot(noise_sequence, mid_em_rmse)
    # plt.legend(['myLs', 'myTLS', 'em'])
    # plt.show()

    if True:
        print("linear : ", mid_linear_rmse)
        print("elastic: ", mid_elastic_rmse)
        print("lasso  : ", mid_lasso_rmse)

        # print("中位数-tls-wb ：", mid_tls_wb)
        # print("中位数-em-wb  ：", mid_em_wb)
        mid_ls_wb = np.array(mid_ls_wb)
        mid_tls_wb = np.array(mid_tls_wb)
        mid_em_wb = np.array(mid_em_wb)
        # print("中位数-tls-wb ：", mid_tls_wb)
        # print("中位数-em-wb  ：", mid_em_wb)

        # 1. 绘制 rmse 图像
        title = "Training VS RMSE\n" + '随机划分数据集次数：' + str(split_num) + '  随机生成噪声次数：' + str(noise_loop)
        x_label = 'Increase of Noise Ratio'
        plotXYs_fn(noise_sequence,
                   [mid_tls_rmse, mid_em_rmse, mid_ls_rmse, mid_linear_rmse, mid_elastic_rmse, mid_lasso_rmse],
                   x_label, 'RMSE', ['tls', 'em', 'ls', 'linear', 'elasticNet', 'lasso'], ['s', 'p', 'o', 'v', '.', '*'],
                   NOW_DIR, 'noise_all.png', title)
        plotXYs_fn(noise_sequence, [mid_tls_rmse, mid_em_rmse], x_label, 'RMSE',
                   ['tls', 'em'], ['s', 'p'], NOW_DIR, 'noise_part.png', title)

        # 2. 绘制 w 和 b 随噪声变化的值。 使用前面的 x_label
        feature_len = len(select_feature)
        plotXWbs_fn(noise_sequence, [mid_tls_wb, mid_em_wb], x_label, ['tls', 'em'], ['s', 'p'],
                    feature_len, NOW_DIR, 'noise_w.png')

        # 3. 保存训练数据  类型+耗时； 数据+w 同；  类型+步长；  两层：划分数据集、随机噪声
        comments = ['噪声比例增大', '耗时：' + str(start) + '=>' + str(end), '特征选择：' + str(select_feature),
                    'hyperparameter：w_dis_epsilon：' + str(w_epsilon) + ',correct:' + str(correct),
                    'noise_pattern ：' + str(noise_pattern),
                    'noise_scale=  ：' + str(noise_min) + ' => ' + str(noise_max) + '(步长' + str(step) + ')',
                    'train_ratio   ：' + str(1 - test_ratio),
                    '随机划分数据集次数：' + str(split_num),
                    '随机生成噪声次数 ：' + str(noise_loop)]
        saveCsvRow_fn(noise_sequence,
                      [mid_tls_rmse, mid_em_rmse, mid_ls_rmse, mid_linear_rmse, mid_elastic_rmse, mid_lasso_rmse,
                       mid_tls_wb.tolist(), mid_em_wb.tolist(), mid_ls_wb.tolist()],
                      'train_ratio',
                      ['tls_rmse', 'em_rmse', 'ls-rmse', 'linear_rmse', 'en_rmse', 'lasso_rmse', 'tls_wb', 'em_wb',
                       'ls_wb'],
                      comments, NOW_DIR, "noise.csv")


if __name__ == '__main__':
    # data_path = 'data/dataset.csv'
    # select_feature = [9, 10, 12, 13, 16]  # 2 3 5 6 9
    data_path = 'data/build_features5.csv'
    select_feature = ['V1/D2/F2', 'F3', 'D5/F5', 'F6', 'F9']  # , 'Area_100_10'
    # 全局使用的变量
    data_x, data_y, convert_y = init_data(data_path, select_feature, 1)
    variable = getconfig('config.json')
    w_epsilon = variable['w_epsilon']
    correct = variable['correct']
    RES_DIR = variable["RES_DIR"]

    only_test = False
    random_seeds = list(range(0, 5))
    for random_id in random_seeds:  # trange(len(random_seeds), desc='Random Process', unit='loop'):
        np.random.seed(random_id)  # random_seeds[random_id]
        noise_pattern = np.random.uniform(0.2, 2, 6)
        # noise_pattern = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        # noise_pattern = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        print(random_id, noise_pattern, "============================================")

        NOW_DIR = os.path.join(RES_DIR, datetime.now().strftime("%Y%m%d%H%M%S"))
        os.makedirs(NOW_DIR)
        if only_test:
            noise_increase(0.0, 0.05, 0.05, test_ratio=0.1, split_num=1, noise_loop=1)
        else:
            noise_increase(0.05, 0.5, 0.05, test_ratio=0.1, split_num=100, noise_loop=50)
    pass
