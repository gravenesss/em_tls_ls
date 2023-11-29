from sklearn.model_selection import train_test_split
from now_utils import *
from util.data_load import init_data, getconfig
from util.ls_tls import tls_fn


# 测试 em 算法
def test_em(test_ratio, cur_seed):
    train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=test_ratio, random_state=cur_seed)
    train_x_std = np.std(train_x, axis=0)
    train_x_mean = np.mean(train_x, axis=0)
    train_y_std = np.std(train_y, axis=0)
    train_y_mean = np.mean(train_y, axis=0)
    train_x_new = (train_x - train_x_mean) / train_x_std
    train_y_new = (train_y - train_y_mean) / train_y_std
    test_x_std = scale(test_x)

    w_std = tls_fn(train_x_new, train_y_new)
    tls_w, tls_b = getWb_fn(train_x.shape[1], w_std, train_x_std, train_x_mean, train_y_std, train_y_mean)
    # train_rmse0 = np.sqrt(mean_squared_error(train_y_new, train_x_new.dot(w_std)))  # 标准化后
    train_rmse1 = np.sqrt(mean_squared_error(train_y, train_x.dot(tls_w) + tls_b))  # wb还原
    train_rmse2 = np.sqrt(mean_squared_error(train_y, train_x_new.dot(w_std) * train_y_std + train_y_mean))  # ms还原
    # test_rmse1_1 = getLossByWb_fn(test_x, test_y, tls_w, tls_b, err_type='rmse')
    test_rmse1_2 = np.sqrt(mean_squared_error(test_y, test_x.dot(tls_w) + tls_b))  # wb还原
    test_rmse2 = np.sqrt(mean_squared_error(test_y, test_x_std.dot(w_std) * train_y_std + train_y_mean))  # ms还原
    print("tls-rmse============================")
    # print("train-标准化：  ", train_rmse0)
    print("train-wb还原:  ", train_rmse1)
    print("train-ms还原:  ", train_rmse2)
    # print("test-wb(fn):", test_rmse1_1)
    print("test-wb还原:   ", test_rmse1_2)
    print("test-ms还原:   ", test_rmse2)

    # em 结果 em中就可以确定是否还原y 计算rmse
    em_err, em_wb = emTest_fn(train_x, train_y, test_x, test_y, w_epsilon, correct, convert_y=convert_y, plot_pic=True)
    print(em_err)

    global count
    if em_err < test_rmse1_2:
        count += 1
        print("yes")

    pass


if __name__ == '__main__':
    # data_path = 'data/dataset.csv'
    # select_feature = [9, 10, 12, 13, 16]  # 2 3 5 6 9
    data_path = 'data/build_features.csv'
    # 'V1/D2/F2', 'D1/F1', 'Area_100_10', 'F6', 'F3', 'F8', 'D3', 'F9', 'F7', 'F4', 'D4', 'D5/F5', 'D6'
    select_feature = ['V1/D2/F2', 'F3', 'F6', 'F9', 'Area_100_10']  # ,   'D5/F5',
    # select_feature = ['V1/D2/F2', 'Area_100_10', 'F6', 'F8', 'D3']  # → F2, Area, F6 F8 D3
    data_x, data_y, convert_y = init_data(data_path, select_feature, 1)  # 全局使用
    variable = getconfig('config.json')
    w_epsilon = variable['w_epsilon']
    correct = variable['correct']

    # 噪声模式固定，查看随机划分数据集 和 随机噪声的影响。
    count = 0
    for i in range(10):
        print(f"seed {i:02d}=================================")
        test_em(0.1, i)
    print(count)

    pass
