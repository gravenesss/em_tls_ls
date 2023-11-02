import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from util.feature_select import *
from util.methods import *
from util.plot_result import *


# 读取变量值：超参数、输出位置
with open("config.json") as f:
    variable = json.load(f)
w_epsilon = variable["w_epsilon"]
correct = variable["correct"]
# RES_DIR = variable["RES_DIR"]
# NOW_DIR = ''
# noise_pattern = np.array([])


# 标准化后的训练集；测试集；标准化前的均值，标准差。
# x_new, y_new, x_test, y_test, x_mean_now, y_mean_now, x_std_now, y_std_now = dataProcess_fn(0.1, True)
def dataProcess_fn(test_size, add_noise):
    # 1）划分数据集
    x, x_test, y, y_test = train_test_split(data_x, data_y, test_size=test_size, random_state=seed)
    if add_noise:
        print("添加随机噪声。")
        noise_ratio = 0.2
        x_ratio = noise_ratio * this_noise_pattern[:-1]  # b.噪声比例 和 c.每一列权重
        y_ratio = noise_ratio * this_noise_pattern[-1]
        x_std_pre = np.std(x, axis=0)  # d. x y 的标准差
        y_std_pre = np.std(y, axis=0)
        new_std_x = np.multiply(x_std_pre, x_ratio)
        new_std_y = np.multiply(y_std_pre, y_ratio)
        np.random.seed(seed)  # a. 随机种子
        x_train_now = x + np.random.normal(0, new_std_x, x.shape)
        y_train_now = y + np.random.normal(0, new_std_y, y.shape)
    else:
        print("不加噪声。")
        x_train_now = x
        y_train_now = y

    # 2）进行标准化
    x_mean_now = np.mean(x_train_now, axis=0)
    y_mean_now = np.mean(y_train_now, axis=0)
    x_std_now = np.std(x_train_now, axis=0)  # em中使用之后的 std
    y_std_now = np.std(y_train_now, axis=0)
    x_new = scale(x_train_now)
    y_new = scale(y_train_now)

    return x_new, y_new, x_test, y_test, x_mean_now, y_mean_now, x_std_now, y_std_now
    pass


# 测试tls: eta为自己设定的值。
def test_tls(eta, add_noise):
    print("TLS Test ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ")
    x_new, y_new, x_test, y_test, x_mean_now, y_mean_now, x_std_now, y_std_now = dataProcess_fn(0.1, add_noise)
    m = 1 if x_new.ndim == 1 else x_new.shape[1]

    # 3）训练：手动设置 diag_x
    diag_x = np.eye(m)
    for i in range(m):
        diag_x[i][i] = eta[i]
    w2 = tls_fn(np.dot(x_new, diag_x), y_new)  # x'=x*sigma_x  w'=sigma_x^(-1)*w  w=sigma_x*w'
    w_std2 = np.dot(diag_x, w2)  # m*m * m*1
    # 还原 w 计算 error
    tls_w2, tls_b2 = getWb_fn(w_std2, y_std_now / x_std_now, m, x_mean_now, y_mean_now)
    tls_err2 = getLossByWb_fn(x_test, y_test, tls_w2, tls_b2, err_type='rmse')

    # 4）输出
    print(eta)
    print("rmse: ", tls_err2)


# 测试 em 算法
def test_em():
    print("EM Test ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ")
    x_new, y_new, x_test, y_test, x_mean_now, y_mean_now, x_std_now, y_std_now = dataProcess_fn(0.1, True)
    m = 1 if x_new.ndim == 1 else x_new.shape[1]

    # 3) 训练
    tls_w_std = tls_fn(x_new, y_new)  # 使用已经进行标准化的数据。tls内部未进行标准化  w未进行还原
    tls_w, tls_b = getWb_fn(tls_w_std, y_std_now / x_std_now, m, x_mean_now, y_mean_now)
    tls_err = getLossByWb_fn(x_test, y_test, tls_w, tls_b, err_type='rmse', convert_y=convert_y)
    # em 结果 em中就可以确定是否还原y 计算rmse
    em_err, em_wb1 = em_fn(x_new, y_new, m, x_std_now, y_std_now, x_mean_now, y_mean_now,
                           x_test, y_test, w_epsilon, correct, convert_y=convert_y)
    # em_err = getLossByWb_fn(x_test, y_test, em_wb1[0:5], em_wb1[5], err_type='rmse', convert_y=convert_y)

    print("tls_err: ", tls_err, "\ntls_wb: ", np.vstack((tls_w, tls_b)))
    print("em_err: ", em_err, "\nem_wb: ", em_wb1)
    pass


data_dir = 'data/dataset.csv'
# this_noise_pattern = np.array([0.95063961, 1.49658409, 0.20020587, 0.74419863, 0.4641606, 0.36620947])
this_noise_pattern = np.array([1.19143622, 1.47466608, 0.72362853, 1.11948969, 1.80730452, 1.81332756])
data_x, data_y = getXy_fn(data_dir, [9, 10, 12, 13, 16])  # 使用更多特征，noise_pattern长度也得是特征个数加1
print("y取以10为底的对数")  # 取不取对数，em效果一致。
# ①注释掉 + convert_y='1' 使用原始数据； ②不注释+convert_y = '1'使用对数数据计算rmse；③ 不注释+convert_y = 'log10'，计算rmse还原
data_y = np.log10(data_y)
convert_y = 'log10'  # 判断是否进行还原，log10 进行还原，1不还原
seed = 42


if __name__ == '__main__':
    # # eta_now = [1.0, 1.0, 1.0, 1.0, 1.0]
    # # eta_now = [0.7, 0.8, 0.9, 0.8, 0.7]
    # eta_now = [1.2, 1.1, 1.0, 1.1, 1.3]
    # test_tls(eta_now, True)

    test_em()

    pass


'''
Tls: 添加随机噪声。
[1.0, 1.0, 1.0, 1.0, 1.0]
rmse:  0.07885943282835027
[0.7, 0.8, 0.9, 0.8, 0.7]
rmse:  0.08164049846471948
[1.2, 1.1, 1.0, 1.1, 1.3]
rmse:  0.07770453843911672
'''
