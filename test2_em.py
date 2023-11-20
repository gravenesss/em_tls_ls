import json
from util.feature_select import *
from now_utils import *

# 读取变量值：超参数、输出位置
with open("config.json") as f:
    variable = json.load(f)
w_epsilon = variable["w_epsilon"]
correct = variable["correct"]


data_path, select_feature, data_x, data_y = None, None, None, None
convert_y = '1'
# 自己提取的数据
def init():
    global data_path, select_feature, data_x, data_y, convert_y
    data_path = 'data/build_feature.csv'
    # ['cell_key', 'D1/F1', 'V1/D2/F2', 'D3', 'D4', 'F3', 'F4', 'D5/F5', 'D6', 'F6', 'F7', 'F8', 'F9', 'cycle_life']
    select_feature = ['V1/D2/F2', 'F3', 'D5/F5', 'F6', 'F9']  # 'V1/D2/F2'
    data_x, data_y = getNewXy_fn(data_path, select_feature)
    print("y取以10为底的对数")  # 取不取对数，em效果一致。
    # ①注释掉 + convert_y='1' 使用原始数据； ②不注释+convert_y = '1'使用对数数据计算rmse；③ 不注释+convert_y = 'log10'，计算rmse还原
    data_y = np.log10(data_y)
    convert_y = '1'  # 判断是否进行还原，log10 进行还原，1不还原
    pass


# 原先数据 dataset.csv
def init1():
    global data_path, select_feature, data_x, data_y, convert_y
    data_path = 'data/dataset.csv'
    select_feature = [9, 10, 12, 13, 16]  # 2 3 5 6 9  select_feature = [9, 10, 12, 14, 16]  # 2 3 5 7 9
    data_x, data_y = getXy_fn(data_path, select_feature)
    print("y取以10为底的对数")
    data_y = np.log10(data_y)
    convert_y = '1'  # 判断是否进行还原，log10 进行还原，1不还原
    pass


# 测试 em 算法
def test_em(cur_seed, add_noise=True):
    print("EM Test ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ")
    train_x, train_y, x_new, y_new, x_test, y_test, x_mean_now, y_mean_now, x_std_now, y_std_now = \
        dataProcess_fn(data_x, data_y, noise_pattern, 0.1, add_noise, cur_seed)
    m = 1 if x_new.ndim == 1 else x_new.shape[1]

    tls_w_std = tls_fn(x_new, y_new)  # 使用已经进行标准化的数据。tls内部未进行标准化  w未进行还原
    tls_w, tls_b = getWb_fn(m, tls_w_std, x_std_now, x_mean_now, y_std_now, y_mean_now)
    # getWb1_fn(tls_w_std, y_std_now/x_std_now, m, x_mean_now, y_mean_now)
    tls_err = getLossByWb_fn(x_test, y_test, tls_w, tls_b, err_type='rmse', convert_y=convert_y)

    # em 结果 em中就可以确定是否还原y 计算rmse
    em_err, em_wb1 = em_fn(train_x, train_y, x_new, y_new, x_std_now, y_std_now, x_mean_now, y_mean_now,
                           x_test, y_test, w_epsilon, correct, convert_y=convert_y)
    # em_err = getLossByWb_fn(x_test, y_test, em_wb1[0:5], em_wb1[5], err_type='rmse', convert_y=convert_y)

    print("tls_err: ", tls_err)
    print("em_err: ", em_err)
    # print("tls_wb: ", np.vstack((tls_w, tls_b)))
    # print("em_wb: ", em_wb1)

    global count
    if em_err < tls_err:
        count += 1

    pass

# 噪声模式固定，查看随机划分数据集 和 随机噪声的影响。 seed_i = 0时效果最好
def test_seed(seeds):
    for i in range(seeds):
        print("seed:", i)
        test_em(i)
    print(count)
    pass

# 测试噪声模式， 划分种子固定，随机噪声种子固定。
def test_pattern(seeds, fixed_seed):
    global noise_pattern
    # 随机划分 和 随机噪声种子固定，随机生成噪声模式
    for i in range(seeds):
        np.random.seed(i)
        this_noise_pattern = np.random.uniform(0.2, 2, 6)
        print("pattern_seed:", i, "pattern:", this_noise_pattern)
        test_em(fixed_seed)

    print(count)
    pass


count = 0


noise_pattern = np.array([0.95063961, 1.49658409, 0.20020587, 0.74419863, 0.4641606, 0.36620947])
# noise_pattern = np.array([1.19143622, 1.47466608, 0.72362853, 1.11948969, 1.80730452, 1.81332756])
# seed = 42

if __name__ == '__main__':
    init()
    test_seed(10)  # 噪声模式固定，查看随机划分数据集 和 随机噪声的影响。
    # todo： 0的趋势和之前的不一样，找找问题； em最小化什么，把目标函数定义出来，看看结果。
    # test_pattern(10, 0)

    pass
