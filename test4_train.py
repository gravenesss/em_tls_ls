import json
from sklearn.preprocessing import scale
from datetime import datetime
from tqdm import trange
from util.feature_select import *
from util.methods import *
from util.plot_result import *
from util.save_read_result import *

# 读取变量值：超参数、输出位置
with open("config.json") as f:
    variable = json.load(f)
w_epsilon = variable["w_epsilon"]
correct = variable["correct"]
RES_DIR = variable["RES_DIR"]
NOW_DIR = ''
noise_pattern = np.array([])


# not modify
def train_data_increase(train_min, train_max, step, noise_ratio, split_num, noise_loop):
    seq_len = int(np.round((train_max + step - train_min) / step))
    train_sequence = [round(x, 3) for x in np.linspace(train_min, train_max + step, seq_len, endpoint=False)]
    data_size = len(data_x)
    test_last_10 = int(data_size * 0.1)  # 未进行四舍五入
    print(train_sequence, len(train_sequence))

    # 0. 记录 tls 和 em 的结果
    mid_tls_rmse, mid_tls_wb = [], []
    mid_em_rmse, mid_em_wb = [], []
    mid_ls_rmse, mid_ls_wb = [], []
    mid_linear_rmse = []
    mid_elastic_rmse = []
    mid_lasso_rmse = []

    start = datetime.now().strftime("%H:%M:%S")
    # 1）训练集比例依次增加
    for train_id in trange(seq_len, desc='Progress', unit='loop'):
        train_size = int(round(data_size * train_sequence[train_id]))  # 四舍五入了
        # print("train_size:", train_size, 'test_size:', test_last_10)
        tmp_tls_rmse, tmp_tls_wb = [], []
        tmp_em_rmse, tmp_em_wb = [], []
        tmp_ls_rmse, tmp_ls_wb = [], []
        tmp_linear_rmse = []
        tmp_elastic_rmse = []
        tmp_lasso_rmse = []

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
                x_ratio = noise_ratio * noise_pattern[:-1]  # b.噪声比例 和 c.每一列权重
                y_ratio = noise_ratio * noise_pattern[-1]
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
                tls_w_std = tls_fn(x_new, y_new)  # 使用已经进行标准化的数据。tls内部未进行标准化  w未进行还原
                tls_w, tls_b = getWb_fn(tls_w_std, y_std_now / x_std_now, m, x_mean_now, y_mean_now)
                tls_err = getLossByWb_fn(x_test, y_test, tls_w, tls_b, err_type='rmse')
                # em 结果
                em_err, em_wb1 = em_fn(x_new, y_new, m, x_std_now, y_std_now, x_mean_now, y_mean_now,
                                       x_test, y_test, w_epsilon, correct)

                # ls
                ls_w_std = ls_fn(x_new, y_new)
                ls_w, ls_b = getWb_fn(ls_w_std, y_std_now / x_std_now, m, x_mean_now, y_mean_now)
                ls_err = getLossByWb_fn(x_test, y_test, ls_w, ls_b, err_type='rmse')

                # C. 记录每次实验的 rmse 和 wb  tls
                tmp_tls_rmse.append(tls_err)
                tmp_tls_wb.append(np.vstack((tls_w, tls_b)).flatten().tolist())
                # em
                tmp_em_rmse.append(em_err)
                tmp_em_wb.append(em_wb1.flatten().tolist())
                # ls
                tmp_ls_rmse.append(ls_err)
                tmp_ls_wb.append(np.vstack((ls_w, ls_b)).flatten().tolist())
                # linear 添加噪声后的数据  convert_y 默认设置为 '1'
                tmp_linear_rmse.append(modelPredict_fn(x_now, y_now, x_test, y_test, 'linear'))
                # elasticNet
                tmp_elastic_rmse.append(modelPredict_fn(x_now, y_now, x_test, y_test, 'en'))
                # Lasso
                tmp_lasso_rmse.append(modelPredict_fn(x_now, y_now, x_test, y_test, 'lasso'))

        pass

        # 记录 随机划分数据集 × 随机噪声 组 的中位数 tls
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
        # ls
        sorted_data = sorted(zip(tmp_ls_rmse, tmp_ls_wb))  # 对应的数据
        mid_index = len(sorted_data) // 2
        mid_err, mid_wb = sorted_data[mid_index]
        mid_ls_rmse.append(mid_err)  # mid_...
        mid_ls_wb.append(mid_wb)  # mid_...
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
    print("tls    : ", mid_tls_rmse)
    print("em     : ", mid_em_rmse)
    print("ls     : ", mid_ls_rmse)
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

    # 1. 绘制 rmse 图像  title、x_label、file_name的前缀 的不一样
    title = "Training VS RMSE\n" + '  随机划分数据集次数：' + str(split_num) + '  随机生成噪声次数：' + str(noise_loop)
    x_label = 'Proportion of Training Data'

    plotXYs_fn(train_sequence,
               [mid_tls_rmse, mid_em_rmse, mid_ls_rmse, mid_linear_rmse, mid_elastic_rmse, mid_lasso_rmse],
               x_label, 'RMSE', ['tls', 'em', 'ls', 'linear', 'elasticNet', 'lasso'], ['s', 'p', 'o', 'v', '.', '*'],
               NOW_DIR, 'train_all.png', title)
    plotXYs_fn(train_sequence, [mid_tls_rmse, mid_em_rmse, mid_ls_rmse], x_label, 'RMSE',
               ['tls', 'em', 'ls'], ['s', 'p', 'o'], NOW_DIR, 'train_part.png', title)

    # 2. 绘制 w 和 b 随噪声变化的值。 使用前面的 x_label
    feature_len = len(select_feature)
    plotXWbs_fn(train_sequence, [mid_tls_wb, mid_em_wb, mid_ls_wb], x_label, ['tls', 'em', 'ls'], ['s', 'p', 'o', ],
                feature_len, NOW_DIR, 'train_w.png')

    # 3. 保存训练数据  类型+耗时； 数据+w 同；  类型+步长；
    comments = ['训练集比例增大', '耗时：' + str(start) + '=>' + str(end), '特征选择：' + str(select_feature),
                'hyperparameter：w_dis_epsilon：' + str(w_epsilon) + ',correct:' + str(correct),
                'noise_pattern ：' + str(noise_pattern),
                'noise_scale   ：' + str(noise_ratio),
                'train_ratio=  ：' + str(train_min) + ' => ' + str(train_max) + '(步长' + str(step) + ')',
                '训练次数：' + str(split_num),
                '随机生成噪声次数：' + str(noise_loop)]
    saveCsvRow_fn(train_sequence,
                  [mid_tls_rmse, mid_em_rmse, mid_ls_rmse, mid_linear_rmse, mid_elastic_rmse, mid_lasso_rmse,
                   mid_tls_wb.tolist(), mid_em_wb.tolist(), mid_ls_wb.tolist()],
                  'train_ratio',
                  ['tls_rmse', 'em_rmse', 'ls-rmse', 'linear_rmse', 'en_rmse', 'lasso_rmse', 'tls_wb', 'em_wb',
                   'ls_wb'],
                  comments, NOW_DIR, "train.csv")


def random_search(random_seeds, only_test=True):
    global noise_pattern
    global RES_DIR
    global NOW_DIR

    if only_test:
        print("onlyTest========================================")
    else:
        print("trueExperiment=================================")

    for random_id in trange(len(random_seeds), desc='Random Process', unit='loop'):
        np.random.seed(random_seeds[random_id])
        noise_pattern = np.random.uniform(0.2, 2, 6)
        # noise_pattern = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        print(random_seeds[random_id], noise_pattern)

        NOW_DIR = os.path.join(RES_DIR, datetime.now().strftime("%Y%m%d%H%M%S"))
        os.makedirs(NOW_DIR)  # 使用 mkdir 函数创建新文件夹
        if only_test:
            train_data_increase(0.2, 0.2, 0.1, noise_ratio=0.2, split_num=100, noise_loop=100)
        else:
            train_data_increase(0.2, 0.9, 0.1, noise_ratio=0.2, split_num=100, noise_loop=100)
    pass


data_path = 'data/build_feature.csv'
# ['cell_key', 'D1/F1', 'V1/D2/F2', 'D3', 'D4', 'F3', 'F4', 'D5/F5', 'D6', 'F6', 'F7', 'F8', 'F9', 'cycle_life']
select_feature = ['V1/D2/F2', 'F3', 'D5/F5', 'F6', 'F9']  # 'V1/D2/F2'
data_x, data_y = getNewXy_fn(data_path, select_feature)
print("y取以10为底的对数")  # 取不取对数，em效果一致。
# ①注释掉 + convert_y='1' 使用原始数据； ②不注释+convert_y = '1'使用对数数据计算rmse；③ 不注释+convert_y = 'log10'，计算rmse还原
data_y = np.log10(data_y)
convert_y = '1'  # 判断是否进行还原，log10 进行还原，1不还原

if __name__ == '__main__':
    random_search(list(range(3, 4)), False)
    pass
