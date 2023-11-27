import json
from now_utils import *
from util.feature_select import getNewXy_fn, getXy_fn
from util.loss import getLossByWb_fn

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


# 测试tls: eta为自己设定的值。
def test_tls(test_ratio, eta):
    train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=test_ratio, random_state=42)
    train_x_std = np.std(train_x, axis=0)
    train_x_mean = np.mean(train_x, axis=0)
    train_y_std = np.std(train_y, axis=0)
    train_y_mean = np.mean(train_y, axis=0)
    train_x_new = (train_x - train_x_mean) / train_x_std
    train_y_new = (train_y - train_y_mean) / train_y_std

    m = train_x_new.shape[1]
    diag_x = np.eye(m)
    for i in range(m):
        diag_x[i][i] = eta[i]

    # tls_fn
    # print("tls_fn========================")
    w1 = tls_fn(train_x_new.dot(diag_x), train_y_new)
    w_std = diag_x.dot(w1)
    tls_w, tls_b = getWb_fn(m, w_std, train_x_std, train_x_mean, train_y_std, train_y_mean)
    # print("w1: ", w1, '\ntls_w', tls_w, '\ntls_b', tls_b)

    train_rmse0 = np.sqrt(mean_squared_error(train_y_new, train_x_new.dot(w_std)))  # 标准化后
    train_rmse1 = np.sqrt(mean_squared_error(train_y, train_x.dot(tls_w) + tls_b))  # wb还原
    train_rmse2 = np.sqrt(mean_squared_error(train_y, train_x_new.dot(w_std) * train_y_std + train_y_mean))  # ms还原
    print("train——rmse=====================")
    print("标准化：  ", train_rmse0)
    print("wb还原:  ", train_rmse1)
    print("ms还原:  ", train_rmse2)

    test_rmse1_1 = getLossByWb_fn(test_x, test_y, tls_w, tls_b, err_type='rmse')
    test_rmse1_2 = np.sqrt(mean_squared_error(test_y, test_x.dot(tls_w) + tls_b))  # wb还原
    test_rmse2 = np.sqrt(mean_squared_error(test_y, test_x.dot(w_std) * train_y_std + train_y_mean))  # ms还原
    print("test——rmse======================")
    print("wb还原:  ", test_rmse1_1)
    print("wb还原:  ", test_rmse1_2)
    print("ms还原:  ", test_rmse2)

    # tls2. ok 两种方法求的w1 一致。
    # print("tls2========================")
    # w1 = tls2(train_x_new.dot(diag_x), train_y_new)
    # print("w1: ", w1)
    # w_std = diag_x.dot(w1)
    # tls_w, tls_b = getWb_fn(m, w_std, train_x_std, train_x_mean, train_y_std, train_y_mean)
    # rmse1 = np.sqrt(mean_squared_error(train_y_new, train_x_new.dot(tls_w) + tls_b))
    # rmse2 = np.sqrt(mean_squared_error(train_y_new, train_x_new.dot(w_std) * train_y_std + train_y_mean))
    # print("original_wb: ", rmse1,
    #       "\nmean_std:    ", rmse2)

    pass


if __name__ == '__main__':
    init()
    noise_pattern = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    # noise_pattern = np.array([0.95063961, 1.49658409, 0.20020587, 0.74419863, 0.4641606, 0.36620947])
    # noise_pattern = np.array([1.19143622, 1.47466608, 0.72362853, 1.11948969, 1.80730452, 1.81332756])
    # print("noise_pattern:", noise_pattern)

    # eta_now = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    # eta_now = np.array([1.21892236e-07, 4.82282049e+00, 4.82282047e+00, 4.82282049e+00, 4.82282050e+00])
    # 1e-8
    eta_now = np.array([2.43739139e-08, 1.00021127e+00, 1.00021126e+00, 1.00021127e+00, 1.00021127e+00])
    # 0.5的会出现traget增大的情况，改变了原来的收敛性
    # eta_now = np.array([0.99787797, 1.41888366, 1.35995768, 1.41529427, 1.41514013])
    print("eta:", eta_now)
    test_tls(0.1, eta_now)

    pass

"""  测试tls: eta为自己设定的值。
使用的特征为： ['V1/D2/F2', 'F3', 'D5/F5', 'F6', 'F9']
y取以10为底的对数
eta: [1. 1. 1. 1. 1.]
train——rmse=====================
标准化：   0.3908244869626299
wb还原:   0.06653896886025107
ms还原:   0.06653896886025097
test——rmse======================
wb还原:   0.2451621469008967
wb还原:   0.2451621469008967
ms还原:   0.6291319334321861

eta: [1.21892236e-07 4.82282049e+00 4.82282047e+00 4.82282049e+00 4.82282050e+00]  # 1e-8
train——rmse=====================
标准化：   0.41520906032733707
wb还原:   0.0706905111046853
ms还原:   0.0706905111046854
test——rmse======================
wb还原:   0.23932813581399606
wb还原:   0.23932813581399606
ms还原:   0.6195988188115839

eta: [0.99787797 1.41888366 1.35995768 1.41529427 1.41514013]  # 0.5 ×
train——rmse=====================
标准化：   0.3871280921281049
wb还原:   0.06590964723636529
ms还原:   0.06590964723636521
test——rmse======================
wb还原:   0.2333831449362848
wb还原:   0.2333831449362848
ms还原:   0.6403400953218694
"""
