from sklearn.preprocessing import scale
from tqdm import trange
from utils import *


# 对于 100 次的数据集划分 生成 50 次的噪声
def noise_increase(noise_min, noise_max, step, test_ratio, split_num, noise_loop):
    seq_len = int(np.round((noise_max + step - noise_min) / step))  # 0.025 ~ 0.5 20个  0.05~0.5 10个
    sequence = [round(x, 3) for x in np.linspace(noise_min, noise_max + step, seq_len, endpoint=False)]
    data_size = len(data_x)
    test_last_10 = int(data_size * test_ratio)  # 未进行四舍五入
    print(sequence, len(sequence))

    # 0. 记录 tls 和 em 的结果
    mid_tls_rmse, mid_tls_wb = [], []
    mid_em_rmse, mid_em_wb = [], []
    mid_linear_rmse = []
    start = datetime.now().strftime("%H:%M:%S")
    # 1）噪声比例以此增大      # for now_id, noise_ratio in enumerate(sequence):
    for now_id in trange(seq_len, desc='Progress', unit='loop'):
        noise_ratio = sequence[now_id]
        tmp_tls_rmse, tmp_tls_wb = [], []
        tmp_em_rmse, tmp_em_wb = [], []
        linear_tmp_rmse = []

        # 2）随机划分数据集
        for split in range(split_num):
            np.random.seed(split)
            random_indices = np.random.permutation(data_size)  # 随机排序
            now_all_x = data_x[random_indices]
            now_all_y = data_y[random_indices]
            # 选取最后 10% 的数据作为 测试集合
            x_test = now_all_x[-test_last_10:]
            y_test = now_all_y[-test_last_10:]
            x = now_all_x[:-test_last_10]
            y = now_all_y[:-test_last_10]
            m = 1 if x.ndim == 1 else x.shape[1]
            x_std_pre = np.std(x, axis=0)  # 噪声标准差使用之前的 std
            y_std_pre = np.std(y, axis=0)

            # 3） 添加 q 组随机噪声。 添加噪声前 pre、添加噪声后 now、标准化后 new。
            for noise_id in range(noise_loop):
                # A. 对训练集添加噪声 并 进行标准化
                x_ratio = noise_ratio * ratio_pattern[:-1]  # b.噪声比例 和 c.每一列权重
                y_ratio = noise_ratio * ratio_pattern[-1]
                new_std_x = np.multiply(x_std_pre, x_ratio)  # d.x y 的标准差
                new_std_y = np.multiply(y_std_pre, y_ratio)
                # 1）对 x y 添加噪声
                np.random.seed(noise_id)  # a.种子不同
                x_now = x + np.random.normal(0, new_std_x, x.shape)
                y_now = y + np.random.normal(0, new_std_y, y.shape)
                # 2）进行标准化
                x_mean_now = np.mean(x_now, axis=0)
                y_mean_now = np.mean(y_now, axis=0)
                x_std_now = np.std(x_now, axis=0)  # em中使用添加噪声之后的 std
                y_std_now = np.std(y_now, axis=0)
                x_new = scale(x_now)
                y_new = scale(y_now)

                # B. 进行训练
                w_new = tls2(x_new, y_new)  # 使用已经进行标准化的数据。tls内部未进行标准化  w未进行还原
                tls_w, tls_b = get_wb(w_new, y_std_now / x_std_now, m, x_mean_now, y_mean_now)
                tls_err, _ = get_rmse_Loss_by_wb(x_test, y_test, tls_w, tls_b)
                # em 结果
                em_err, em_wb1 = em_tls(x_new, y_new, m, x_std_now, y_std_now, x_mean_now, y_mean_now,
                                        x_test, y_test, w_dis_epsilon)

                # C. 记录每次实验的 rmse 和 wb
                # [np.array([[-4.39078963e-01], [2.21002629e-04], [-1.85588998e+00], [-3.83108998e+00]])]
                # 转为[-4.39078963e-01, 2.21002629e-04, -1.85588998e+00, -3.83108998e+00]
                # print(em_wb1)
                tmp_tls_rmse.append(tls_err)
                tmp_tls_wb.append(np.vstack((tls_w, tls_b)).flatten().tolist())
                tmp_em_rmse.append(em_err)
                tmp_em_wb.append(em_wb1.flatten().tolist())

                # linear 添加噪声后的数据
                linear_tmp_rmse.append(model_rmse(x_now, y_now, x_test, y_test, 'linear'))

        # 记录 随机划分数据集 × 随机噪声 组 的中位数
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
        linear_tmp_rmse.sort(reverse=False)
        mid_linear_rmse.append(linear_tmp_rmse[len(linear_tmp_rmse) // 2])

    end = datetime.now().strftime("%H:%M:%S")
    print(start + " -- " + end)
    print("中位数-tls   ：", mid_tls_rmse)
    print("中位数-tls em：", mid_em_rmse)
    print("中位数-linear:", mid_linear_rmse)
    # print("中位数-tls-wb ：", mid_tls_wb)
    # print("中位数-em-wb  ：", mid_em_wb)
    mid_tls_wb = np.array(mid_tls_wb)  # 之前写错了mid_tls_rmse
    mid_em_wb = np.array(mid_em_wb)
    # print("中位数-tls-wb ：", mid_tls_wb)
    # print("中位数-em-wb  ：", mid_em_wb)

    # 1. 绘制 rmse 图像
    title = "Training VS RMSE\n" + ' 噪声比例：' + str(noise_min) + ' => ' + str(noise_max) + '(步长' + str(
        step) + ')\n' + '  随机划分数据集次数：' + str(split_num) + '  随机生成噪声次数：' + str(noise_loop)
    x_label = 'Increase of Noise Ratio'
    # plot_xyy(sequence, mid_tls_rmse, mid_em_rmse, title, 'RMSE', x_label, NOW_DIR, 'noise.png')
    plot_xys(sequence, [mid_tls_rmse, mid_em_rmse, mid_linear_rmse],
             ['tls', 'em', 'linear'], ['o', 'v', '.'], x_label, 'RMSE', NOW_DIR, 'noise.png')

    # 2. 绘制 w 和 b 随噪声变化的值。 使用前面的 x_label
    feature_len = len(select_feature)
    # plot_xwb(sequence, mid_tls_wb, mid_em_wb, x_label, feature_len, NOW_DIR, 'noise_w.png')
    plot_x_wbs(sequence, [mid_tls_wb, mid_em_wb], ['tls', 'em'], ['o', 'v'],
               x_label, feature_len, NOW_DIR, 'train_w.png')

    # 3. 保存训练数据  类型+耗时； 数据+w 同；  类型+步长；  两层：划分数据集、随机噪声
    comments = ['噪声比例增大', '耗时：' + str(start) + '=>' + str(end),
                '使用的数据为：' + feature_select,
                '前后两次 w 的差距：' + str(w_dis_epsilon),
                '噪声模式为：' + str(ratio_pattern),
                '噪声比例：' + str(noise_min) + ' => ' + str(noise_max) + '(步长' + str(step) + ')',
                '随机划分数据集次数：' + str(split_num) + '',
                '随机生成噪声次数：' + str(noise_loop) + '']
    # save_csv(sequence, mid_tls_rmse, mid_em_rmse, mid_tls_wb.tolist(), mid_em_wb.tolist(),
    #          comments, NOW_DIR, "noise.csv")
    save_csvs(sequence, [mid_tls_rmse, mid_em_rmse, mid_tls_wb.tolist(), mid_em_wb.tolist(), mid_linear_rmse],
              'noise_ratio', ['tls_rmse', 'em_rmse', 'tls_wb', 'em_wb', 'linear_rmse'], comments, NOW_DIR, "noise.csv")


# 训练集比例依次增大，但是每组只使用最后10%作为测试集  测试次数，训练比例，随机噪声
def train_data_increase(train_min, train_max, step, noise_ratio, split_num, noise_loop):
    seq_len = int(np.round((train_max + step - train_min) / step))
    sequence = [round(x, 3) for x in np.linspace(train_min, train_max + step, seq_len, endpoint=False)]
    data_size = len(data_x)
    test_last_10 = int(data_size * 0.1)  # 未进行四舍五入
    print(sequence, len(sequence))

    # 0. 记录 tls 和 em 的结果
    mid_tls_rmse, mid_tls_wb = [], []
    mid_em_rmse, mid_em_wb = [], []
    mid_linear_rmse = []
    mid_ls_rmse, mid_ls_wb = [], []

    start = datetime.now().strftime("%H:%M:%S")
    # 1）训练集比例依次增加
    for train_id in trange(seq_len, desc='Progress', unit='loop'):
        train_size = int(round(data_size * sequence[train_id]))  # 四舍五入了
        # print("train_size:", train_size, 'test_size:', test_last_10)
        tmp_tls_rmse, tmp_tls_wb = [], []
        tmp_em_rmse, tmp_em_wb = [], []
        linear_tmp_rmse = []
        tmp_ls_rmse, tmp_ls_wb = [], []

        # 2）随机划分数据集集
        for split in range(split_num):
            np.random.seed(split)
            random_indices = np.random.permutation(data_size)  # 随机排序
            now_all_x = data_x[random_indices]
            now_all_y = data_y[random_indices]
            # 选取最后 10% 的数据作为 测试集合
            x_test = now_all_x[-test_last_10:]
            y_test = now_all_y[-test_last_10:]
            # 选取前 ratio 的数据作为训练集  有区别
            x = now_all_x[:train_size]
            y = now_all_y[:train_size]
            m = 1 if x.ndim == 1 else x.shape[1]
            x_std_pre = np.std(x, axis=0)  # 噪声标准差使用之前的 std
            y_std_pre = np.std(y, axis=0)

            # 3） 添加 q 组随机噪声。 添加噪声前 pre、添加噪声后 now、标准化后 new。
            for noise_id in range(noise_loop):
                # A. 对训练集添加噪声 并 进行标准化
                x_ratio = noise_ratio * ratio_pattern[:-1]  # b.噪声比例 和 c.每一列权重
                y_ratio = noise_ratio * ratio_pattern[-1]
                new_std_x = np.multiply(x_std_pre, x_ratio)  # d.x y 的标准差
                new_std_y = np.multiply(y_std_pre, y_ratio)
                # 1）对 x y 添加噪声
                np.random.seed(noise_id)  # a.种子不同
                x_now = x + np.random.normal(0, new_std_x, x.shape)
                y_now = y + np.random.normal(0, new_std_y, y.shape)
                # 2）进行标准化
                x_mean_now = np.mean(x_now, axis=0)
                y_mean_now = np.mean(y_now, axis=0)
                x_std_now = np.std(x_now, axis=0)  # em中使用添加噪声之后的 std
                y_std_now = np.std(y_now, axis=0)
                x_new = scale(x_now)
                y_new = scale(y_now)

                # B. 进行训练
                tls_w_std = tls2(x_new, y_new)  # 使用已经进行标准化的数据。tls内部未进行标准化  w未进行还原
                tls_w, tls_b = get_wb(tls_w_std, y_std_now / x_std_now, m, x_mean_now, y_mean_now)
                tls_err, _ = get_rmse_Loss_by_wb(x_test, y_test, tls_w, tls_b)
                # em 结果
                em_err, em_wb1 = em_tls(x_new, y_new, m, x_std_now, y_std_now, x_mean_now, y_mean_now,
                                        x_test, y_test, w_dis_epsilon)
                # ls
                ls_w_std = ls(x_new, y_new)
                ls_w, ls_b = get_wb(ls_w_std, y_std_now / x_std_now, m, x_mean_now, y_mean_now)
                ls_err, _ = get_rmse_Loss_by_wb(x_test, y_test, ls_w, ls_b)

                # C. 记录每次实验的 rmse
                tmp_tls_rmse.append(tls_err)
                tmp_tls_wb.append(np.vstack((tls_w, tls_b)).flatten().tolist())
                tmp_em_rmse.append(em_err)
                tmp_em_wb.append(em_wb1.flatten().tolist())

                # linear 添加噪声后的数据
                linear_tmp_rmse.append(model_rmse(x_now, y_now, x_test, y_test, 'linear'))

                # ls
                tmp_ls_rmse.append(ls_err)
                tmp_ls_wb.append(np.vstack((ls_w, ls_b)).flatten().tolist())

        # 记录 随机划分数据集 × 随机噪声 组 的中位数
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
        linear_tmp_rmse.sort(reverse=False)
        mid_linear_rmse.append(linear_tmp_rmse[len(linear_tmp_rmse) // 2])

        # ls
        sorted_data = sorted(zip(tmp_ls_rmse, tmp_ls_wb))  # 对应的数据
        mid_index = len(sorted_data) // 2
        mid_err, mid_wb = sorted_data[mid_index]
        mid_ls_rmse.append(mid_err)     # mid_...
        mid_ls_wb.append(mid_wb)        # mid_...

    end = datetime.now().strftime("%H:%M:%S")
    print(start + " -- " + end)
    print("中位数-tls with noise：", mid_tls_rmse)
    print("中位数-tls em：", mid_em_rmse)
    print("中位数-linear:", mid_linear_rmse)
    print("中位数-ls    :", mid_ls_rmse)
    # print("中位数-tls-wb ：", mid_tls_wb)
    # print("中位数-em-wb  ：", mid_em_wb)
    mid_tls_wb = np.array(mid_tls_wb)  # 之前写错了mid_tls_rmse
    mid_em_wb = np.array(mid_em_wb)
    mid_ls_wb = np.array(mid_ls_wb)
    # print("中位数-tls-wb ：", mid_tls_wb)
    # print("中位数-em-wb  ：", mid_em_wb)

    # 1. 绘制 rmse 图像  title、x_label、file_name的前缀 的不一样
    title = "Training VS RMSE\n" + '  训练集比例：' + str(train_min) + ' => ' + str(train_max) + '(步长' + str(
        step) + ')\n' + '  随机划分数据集次数：' + str(split_num) + '  随机生成噪声次数：' + str(noise_loop)
    x_label = 'Proportion of Training Data'
    # plot_xyy(sequence, mid_tls_rmse, mid_em_rmse, title, 'RMSE', x_label, NOW_DIR, 'train.png')
    plot_xys(sequence, [mid_tls_rmse, mid_em_rmse, mid_linear_rmse, mid_ls_rmse],
             ['tls', 'em', 'linear', 'ls'], ['o', 'v', '.', '*'], x_label, 'RMSE', NOW_DIR, 'train.png')

    # 2. 绘制 w 和 b 随噪声变化的值。 使用前面的 x_label. 保存图片的名字变化了。
    feature_len = len(select_feature)
    # plot_xwb(sequence, mid_tls_wb, mid_em_wb, x_label, feature_len, NOW_DIR, 'train_w.png')
    plot_x_wbs(sequence, [mid_tls_wb, mid_em_wb, mid_ls_wb],
               ['tls', 'em', 'ls'], ['o', 'v', '*'], x_label, feature_len, NOW_DIR, 'train_w.png')

    # 3. 保存训练数据  类型+耗时； 数据+w 同；  类型+步长；
    comments = ['训练集比例增大', '耗时：' + str(start) + '=>' + str(end),
                '使用的数据为：' + feature_select,
                '前后两次 w 的差距: ' + str(w_dis_epsilon),
                '噪声比例：' + str(noise_ratio),
                '噪声模式为：' + str(ratio_pattern),
                '训练次数：' + str(split_num) + '',
                '训练集比例：' + str(train_min) + ' => ' + str(train_max) + '(步长' + str(step) + ')',
                '随机生成噪声次数：' + str(noise_loop) + '']
    # save_csv(sequence, mid_tls_rmse, mid_em_rmse, mid_tls_wb.tolist(), mid_em_wb.tolist(),
    #          comments, NOW_DIR, "train.csv")
    save_csvs(sequence, [mid_tls_rmse, mid_em_rmse, mid_tls_wb.tolist(), mid_em_wb.tolist(), mid_linear_rmse],
              'train_ratio', ['tls_rmse', 'em_rmse', 'tls_wb', 'em_wb', 'linear_rmse'], comments, NOW_DIR, "train.csv")


def random_search(random_seeds, only_test=True, exp_type=0):
    global ratio_pattern
    global RES_DIR
    global NOW_DIR

    for random_id in trange(len(random_seeds), desc='Random Process', unit='loop'):
        np.random.seed(random_seeds[random_id])
        ratio_pattern = np.random.uniform(0.2, 2, 6)
        # ratio_pattern = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        # print(random_seeds[random_id], ratio_pattern)

        NOW_DIR = os.path.join(RES_DIR, datetime.now().strftime("%Y%m%d%H%M%S"))
        os.makedirs(NOW_DIR)  # 使用 mkdir 函数创建新文件夹
        if only_test:
            print("onlyTest========================================")
            if exp_type == 0:
                print("噪声 比例依次增大")
                noise_increase(0.2, 0.225, 0.025, test_ratio=0.1, split_num=100, noise_loop=100)
            elif exp_type == 1:
                print("训练集 比例依次增大")
                train_data_increase(0.85, 0.9, 0.05, noise_ratio=0.1, split_num=100, noise_loop=100)
            else:
                print("噪声 比例依次增大")
                noise_increase(0.2, 0.225, 0.025, test_ratio=0.1, split_num=100, noise_loop=100)
                print("训练集 比例依次增大")
                train_data_increase(0.85, 0.9, 0.05, noise_ratio=0.1, split_num=100, noise_loop=100)
        else:
            print("trueExperiment=================================")
            if exp_type == 0:
                print("噪声 比例依次增大")
                noise_increase(0.05, 0.5, 0.05, test_ratio=0.1, split_num=100, noise_loop=100)
            elif exp_type == 1:
                print("训练集 比例依次增大")
                train_data_increase(0.20, 0.90, 0.1, noise_ratio=0.1, split_num=100, noise_loop=100)
            else:
                print("噪声 比例依次增大")
                noise_increase(0.05, 0.5, 0.05, test_ratio=0.1, split_num=100, noise_loop=100)
                print("训练集 比例依次增大")
                train_data_increase(0.20, 0.90, 0.1, noise_ratio=0.1, split_num=100, noise_loop=100)


# 训练其他模型
def model_test():
    train_min, train_max, step = 0.2, 0.9, 0.1
    split_num = 100

    seq_len = int(np.round((train_max + step - train_min) / step))
    sequence = [round(x, 3) for x in np.linspace(train_min, train_max + step, seq_len, endpoint=False)]
    data_size = len(data_x)
    test_last_10 = int(data_size * 0.1)  # 未进行四舍五入
    print(sequence, len(sequence))

    linear_mid_rmse = []  # 线性回归
    tree_mid_rmse = []  # 决策树
    svr_mid_rmse = []  # 支持向量机
    rf_mid_rmse = []  # 随机森林
    knn_mid_rmse = []  # k近邻
    poly_mid_rmse = []  # 多项式回归
    en_mid_rmse = []  # 弹性网络

    # 1）训练集比例依次增加
    for train_id in trange(seq_len, desc='Progress', unit='loop'):
        train_size = int(round(data_size * sequence[train_id]))  # 四舍五入了
        # print("train_size:", train_size, 'test_size:', test_last_10)
        linear_tmp_rmse = []
        tree_tmp_rmse = []
        svr_tmp_rmse = []
        rf_tmp_rmse = []
        knn_tmp_rmse = []
        poly_tmp_rmse = []
        en_tmp_rmse = []

        # 2）随机划分数据集集
        for split in range(split_num):
            # 1. 划分数据集训练集
            np.random.seed(split)
            random_indices = np.random.permutation(data_size)  # 随机排序
            now_all_x = data_x[random_indices]
            now_all_y = data_y[random_indices]
            # 选取最后 10% 的数据作为 测试集合
            x_test = now_all_x[-test_last_10:]
            y_test = now_all_y[-test_last_10:]
            # 选取前 ratio 的数据作为训练集  有区别
            x = now_all_x[:train_size]
            y = now_all_y[:train_size]

            # 记录每次实验的 rmse
            linear_tmp_rmse.append(model_rmse(x, y, x_test, y_test, 'linear'))
            tree_tmp_rmse.append(model_rmse(x, y, x_test, y_test, 'tree'))
            svr_tmp_rmse.append(model_rmse(x, y, x_test, y_test, 'svr'))
            rf_tmp_rmse.append(model_rmse(x, y, x_test, y_test, 'rf'))
            knn_tmp_rmse.append(model_rmse(x, y, x_test, y_test, 'knn'))
            poly_tmp_rmse.append(model_rmse(x, y, x_test, y_test, 'poly', poly=3))
            en_tmp_rmse.append(model_rmse(x, y, x_test, y_test, 'en'))

        # tmp_rmse = sorted(tmp_rmse, reverse=False)  # False 升序 True 降序
        linear_tmp_rmse.sort(reverse=False)
        linear_mid_rmse.append(linear_tmp_rmse[len(linear_tmp_rmse) // 2])

        tree_tmp_rmse.sort(reverse=False)
        tree_mid_rmse.append(tree_tmp_rmse[len(tree_tmp_rmse) // 2])

        svr_tmp_rmse.sort(reverse=False)
        svr_mid_rmse.append(svr_tmp_rmse[len(svr_tmp_rmse) // 2])

        rf_tmp_rmse.sort(reverse=False)
        rf_mid_rmse.append(rf_tmp_rmse[len(rf_tmp_rmse) // 2])

        knn_tmp_rmse.sort(reverse=False)
        knn_mid_rmse.append(knn_tmp_rmse[len(knn_tmp_rmse) // 2])

        poly_tmp_rmse.sort(reverse=False)
        poly_mid_rmse.append(poly_tmp_rmse[len(poly_tmp_rmse) // 2])

        en_tmp_rmse.sort(reverse=False)
        en_mid_rmse.append(en_tmp_rmse[len(en_tmp_rmse) // 2])

    print('linear : ', linear_mid_rmse)
    print('tree   : ', tree_mid_rmse)
    print('svr    : ', svr_mid_rmse)
    print('rf     : ', rf_mid_rmse)
    print('knn    : ', knn_mid_rmse)
    print('poly   : ', poly_mid_rmse)
    print('elastic: ', en_mid_rmse)

    # 绘制图像
    plt.rcParams['font.family'] = ['SimSun']
    plt.plot(sequence, linear_mid_rmse, label='linear', marker='o')
    plt.plot(sequence, tree_mid_rmse, label='tree', marker='v')
    plt.plot(sequence, svr_mid_rmse, label='svr', marker='^')
    plt.plot(sequence, rf_mid_rmse, label='rf', marker='s')
    plt.plot(sequence, knn_mid_rmse, label='knn', marker='p')
    # plt.plot(sequence, poly_mid_rmse, label='poly', marker='*')
    plt.plot(sequence, en_mid_rmse, label='elastic', marker='d')

    plt.xlabel('Proportion of Training Data')
    plt.ylabel('RMSE')
    plt.legend()
    now_time = datetime.now().strftime('%Y%m%d%H%M%S')
    plt.savefig('result/model/'+now_time+'.png')
    plt.show()

    yy = [linear_mid_rmse, tree_mid_rmse, svr_mid_rmse, rf_mid_rmse, knn_mid_rmse, poly_mid_rmse, en_mid_rmse]
    xx_label = ['linear', 'tree', 'svr', 'rf', 'knn', 'poly', 'elastic']
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    for i, ax in enumerate(axes.flatten()):
        if i == 7:
            break
        ax.plot(sequence, yy[i], label=xx_label[i], marker='o')
        ax.set_xlabel('Proportion of Training Data')
        ax.set_ylabel('RMSE')
        ax.legend()
    plt.tight_layout()
    plt.savefig('result/model/model_all.png')
    plt.show()

    # linear :  [0.06233413980722508, 0.0612720290801215, 0.061927052739399316, 0.06097482964532715, 0.06449011550872968, 0.06400341851491136, 0.06398068602275588, 0.06480867771290019]
    # tree   :  [0.09748936293269787, 0.08829232751905891, 0.08581751954877728, 0.08024031589778116, 0.07664961072413928, 0.07152474497781357, 0.07111230607201156, 0.07081783263467839]
    # svr    :  [0.1804230264715743, 0.1751433944005729, 0.1735490629197158, 0.1737380070268008, 0.17344362514096992, 0.17334898385826625, 0.1731811163646808, 0.17236088251988313]
    # rf     :  [0.07753955037330654, 0.06821163372908828, 0.06418785376999427, 0.06131932566051015, 0.05951572225857085, 0.05494756439259129, 0.05134177351728498, 0.05054837473720021]
    # knn    :  [0.18848325415031303, 0.1789168883404452, 0.18017030493878214, 0.1764429168845137, 0.1762000556153448, 0.17516645310049622, 0.17459994208905452, 0.18176054541923037]
    # poly   :  [1.096792831260081, 0.3119281229115725, 0.15873804842896908, 0.17460710072725827, 0.12175049493914726, 0.10430912950601386, 0.09503499745299557, 0.08123947271002911]
    # elastic:  [0.17984211959251162, 0.1762369131762991, 0.1717515996037016, 0.1756460303045435, 0.174042955851933, 0.1747027259894932, 0.1741177143790075, 0.17245865722740497]

    pass


RES_DIR = 'result'
NOW_DIR = ''
w_dis_epsilon = 1e-6

# select_feature = [9, 10, 12, 13, 16]   # 2 3 5 6 9
select_feature = [9, 10, 12, 14, 16]  # 2 3 5 7 9  F1+7   8:17 F1到F9
feature_select = ' '.join(str(x) for x in select_feature)
data_x, data_y = get_feature(select_feature)
data_y = np.log10(data_y)  # log10  可以使得y关于x的函数是非线性的  10 50 → 100 100

ratio_pattern = np.array([])
# noise: -0.475, -50 3 0.01  -5  -2  → /50 0.008  1  0.06  0.0002  0.04
# ratio_pattern = np.array([0.06, 0.0002, 0.008, 1,  0.04, 0.65])  # 每列噪声缩放比例 按照特征的 w 顺序
# ratio_pattern = np.array([0.008, 1, 0.06, 0.0002, 0.04, 0.65])  # 每列噪声缩放比例 按照特征的 w 降序

# train: -0.48 -100 3 0.01  25  -2  → 0.005 1  0.03  0.00001  0.25
# ratio_pattern = np.array([0.03, 0.00001, 0.005, 1, 0.25, 0.65])
# ratio_pattern = np.array([0.005, 1, 0.03, 0.00001, 0.25, 0.65])

# 随机
# ratio_pattern = np.array([0.6, 0.9, 0.8, 1.0, 0.7, 0.65])  # 每列噪声缩放比例 # 训练not ok


if __name__ == '__main__':
    # test_em()
    random_search(list(range(1, 5)), False, 0)
    # model_test()
    pass
