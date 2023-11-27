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
def init():  # 2 3 6 area
    global data_path, select_feature, data_x, data_y, convert_y
    data_path = 'data/rebuild_features5.csv'
    # ['cell_key', 'D1/F1', 'V1/D2/F2', 'D3', 'D4', 'F3', 'F4', 'D5/F5', 'D6', 'F6', 'F7', 'F8', 'F9', 'cycle_life']
    select_feature = ['V1/D2/F2', 'F3', 'D5/F5', 'F6', 'F9', 'Area_100_10']  # 'V1/D2/F2'
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
def test_em(test_ratio, cur_seed):
    train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=test_ratio, random_state=cur_seed)
    train_x_std = np.std(train_x, axis=0)
    train_x_mean = np.mean(train_x, axis=0)
    train_y_std = np.std(train_y, axis=0)
    train_y_mean = np.mean(train_y, axis=0)
    train_x_new = (train_x - train_x_mean) / train_x_std
    train_y_new = (train_y - train_y_mean) / train_y_std
    m = train_x_new.shape[1]

    # tls
    tls_wb = tls_fn(train_x_new, train_y_new)  # 使用已经进行标准化的数据。tls内部未进行标准化  w未进行还原
    tls_w, tls_b = getWb_fn(m, tls_wb, train_x_std, train_x_mean, train_y_std, train_y_mean)
    train_rmse1 = np.sqrt(mean_squared_error(train_y, train_x.dot(tls_w) + tls_b))  # wb还原
    train_rmse2 = np.sqrt(mean_squared_error(train_y, train_x_new.dot(tls_wb) * train_y_std + train_y_mean))  # ms还原
    print("train——rmse=====================")
    # print("标准化：  ", train_rmse0)
    print("wb还原:  ", train_rmse1)
    print("ms还原:  ", train_rmse2)
    test_rmse1 = np.sqrt(mean_squared_error(test_y, test_x.dot(tls_w) + tls_b))  # wb还原
    test_rmse2 = np.sqrt(mean_squared_error(test_y, test_x.dot(tls_wb) * train_y_std + train_y_mean))  # ms还原
    print("test——rmse======================")
    print("wb还原:  ", test_rmse1)
    print("ms还原:  ", test_rmse2)

    # em 结果 em中就可以确定是否还原y 计算rmse  train_x, train_y,
    em_err, em_wb = em_fn(train_x, train_y, test_x, test_y, train_x_new, train_y_new, train_x_std, train_y_std,
                          train_x_mean, train_y_mean, w_epsilon, correct, convert_y=convert_y)
    print(em_err)

    global count
    if em_err < test_rmse1:
        count += 1

    pass


# 查看随机划分数据集 和 随机噪声的影响。
def test_seed(seeds):
    for i in range(seeds):
        print("seed:", i)
        test_em(0.1, i)
    print(count)
    pass


count = 0


if __name__ == '__main__':
    init()
    test_seed(10)  # 噪声模式固定，查看随机划分数据集 和 随机噪声的影响。

    pass
