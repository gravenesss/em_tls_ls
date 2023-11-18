import json
from now_utils import *
from util.feature_select import getNewXy_fn, getXy_fn

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
def test_tls(eta, noise_pattern_, add_noise):
    print("TLS Test ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ")
    x_new, y_new, x_test, y_test, x_mean_now, y_mean_now, x_std_now, y_std_now = \
        dataProcess_fn(data_x, data_y, noise_pattern_, 0.1, add_noise)
    m = 1 if x_new.ndim == 1 else x_new.shape[1]

    diag_x = np.eye(m)
    for i in range(m):
        diag_x[i][i] = eta[i]

    # 计算 w
    w2 = tls_fn(np.dot(x_new, diag_x), y_new)  # x'=x*sigma_x  w'=sigma_x^(-1)*w  w=sigma_x*w'
    w_std2 = np.dot(diag_x, w2)  # m*m * m*1
    # 还原 w 计算 error  getWb_fn(m, w_std, x_std, x_mean, y_std, y_mean)
    tls_w2, tls_b2 = getWb_fn(m, w_std2, x_std_now, x_mean_now, y_std_now, y_mean_now)
    # getWb_fn(w_std2, y_std_now / x_std_now, m, x_mean_now, y_mean_now)
    tls_err2 = getLossByWb_fn(x_test, y_test, tls_w2, tls_b2, err_type='rmse')

    # 4）输出
    print("rmse: ", tls_err2)


if __name__ == '__main__':
    init()
    # noise_pattern = np.array([1.19143622, 1.47466608, 0.72362853, 1.11948969, 1.80730452, 1.81332756])
    # print("对角矩阵已知(不全为1)， 噪声模式不全为1：")
    # noise_pattern = np.array([0.95063961, 1.49658409, 0.20020587, 0.74419863, 0.4641606, 0.36620947])
    # eta_now = np.array([1.21892236e-07, 4.82282049e+00, 4.82282047e+00, 4.82282049e+00, 4.82282050e+00])

    # noise_pattern = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    noise_pattern = np.array([0.95063961, 1.49658409, 0.20020587, 0.74419863, 0.4641606, 0.36620947])
    eta_now = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

    print("noise_pattern:", noise_pattern)
    print("eta:", eta_now)
    test_tls(eta_now, noise_pattern, True)

    pass

"""
使用的特征为： ['V1/D2/F2', 'F3', 'D5/F5', 'F6', 'F9']
y取以10为底的对数
对角矩阵已知(不全为1)， 噪声模式不全为1：
noise_pattern: [0.95063961 1.49658409 0.20020587 0.74419863 0.4641606  0.36620947]
eta: [1.21892236e-07 4.82282049e+00 4.82282047e+00 4.82282049e+00 4.82282050e+00]
TLS Test ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== 
添加随机噪声。
rmse:  0.20479057509073972

对角矩阵全1， 噪声模式全1
noise_pattern: [1. 1. 1. 1. 1. 1.]
eta: [1. 1. 1. 1. 1.]
TLS Test ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== 
添加随机噪声。
rmse:  0.212130202443861

noise_pattern: [0.95063961 1.49658409 0.20020587 0.74419863 0.4641606  0.36620947]
eta: [1. 1. 1. 1. 1.]
TLS Test ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== 
添加随机噪声。
rmse:  0.20829796391253924
"""

