from sklearn.model_selection import train_test_split
from now_utils import *
from util.data_load import init_data
from util.methods import tls_fn
from util.loss import getLossByWb_fn


# 测试tls: eta为自己设定的值。
def test_tls(test_ratio, eta):
    train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=test_ratio, random_state=42)
    train_x_std = np.std(train_x, axis=0)
    train_x_mean = np.mean(train_x, axis=0)
    train_y_std = np.std(train_y, axis=0)
    train_y_mean = np.mean(train_y, axis=0)
    train_x_new = (train_x - train_x_mean) / train_x_std
    train_y_new = (train_y - train_y_mean) / train_y_std

    m = train_x.shape[1]
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
    test_rmse1_1 = getLossByWb_fn(test_x, test_y, tls_w, tls_b, err_type='rmse')
    test_rmse1_2 = np.sqrt(mean_squared_error(test_y, test_x.dot(tls_w) + tls_b))  # wb还原
    test_rmse2 = np.sqrt(mean_squared_error(test_y, test_x.dot(w_std) * train_y_std + train_y_mean))  # ms还原
    print("rmse==========================")
    print("train-标准化：  ", train_rmse0)
    print("train-wb还原:  ", train_rmse1)
    print("train-ms还原:  ", train_rmse2)
    print("test-wb还原:   ", test_rmse1_1)
    print("test-wb还原(fn)", test_rmse1_2)
    print("test-ms还原:   ", test_rmse2)

    pass


if __name__ == '__main__':
    # data_path = 'data/dataset.csv'
    # select_feature = [9, 10, 12, 13, 16]  # 2 3 5 6 9
    data_path = 'data/build_features.csv'
    # 'V1/D2/F2', 'D1/F1', 'Area_100_10', 'F6', 'F3', 'F8', 'D3', 'F9', 'F7', 'F4', 'D4', 'D5/F5', 'D6'
    select_feature = ['V1/D2/F2', 'Area_100_10', 'F6', 'F8', 'D3']  # → F2, Area, F6 F8 D3
    data_x, data_y, convert_y = init_data(data_path, select_feature, 1)  # 全局使用

    # eta_now = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    # eta_now = np.array([1.21892236e-07, 4.82282049e+00, 4.82282047e+00, 4.82282049e+00, 4.82282050e+00])
    eta_now = np.array([2.43739139e-08, 1.00021127e+00, 1.00021126e+00, 1.00021127e+00, 1.00021127e+00])  # 1e-8
    print("eta:", eta_now)
    test_tls(0.1, eta_now)

    pass

"""  测试tls: eta为自己设定的值。
使用的特征为： ['V1/D2/F2', 'F3', 'D5/F5', 'F6', 'F9']
y取以10为底的对数
eta: [1. 1. 1. 1. 1.]
rmse==========================
train-标准化：   0.3908244869626299
train-wb还原:   0.06653896886025107
train-ms还原:   0.06653896886025097
test-wb还原:    0.2451621469008967
test-wb还原(fn) 0.2451621469008967
test-ms还原:    0.6291319334321861

eta: [2.43739139e-08 1.00021127e+00 1.00021126e+00 1.00021127e+00 1.00021127e+00]
rmse==========================
train-标准化：   0.41520906032734467
train-wb还原:   0.07069051110468667
train-ms还原:   0.07069051110468666
test-wb还原:    0.2393281358140008
test-wb还原(fn) 0.2393281358140008
test-ms还原:    0.6195988188115784
"""
