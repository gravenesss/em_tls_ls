from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from utils import *


# 没有加噪声
if __name__ == '__main__':
    select_feature = [9, 10, 12, 13, 16]  # 2 3 5 6 9  select_feature = [9, 10, 12, 14, 16]  # 2 3 5 7 9
    data_x, data_y = get_feature(select_feature)
    data_y = np.log10(data_y)  # log10  可以使得y关于x的函数是非线性的  10 50 → 100 100
    add_noise = True
    seed = 42

    # 1）划分数据集
    x, x_test, y, y_test = train_test_split(data_x, data_y, test_size=0.1, random_state=seed)
    m = 1 if x.ndim == 1 else x.shape[1]
    if add_noise:
        print("添加随机噪声。")
        this_noise_pattern = np.array([0.95063961, 1.49658409, 0.20020587, 0.74419863, 0.4641606, 0.36620947])
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

    # 3）训练
    tls_w_std = tls2(x_new, y_new)  # 使用已经进行标准化的数据。tls内部未进行标准化  w未进行还原
    tls_w, tls_b = get_wb(tls_w_std, y_std_now / x_std_now, m, x_mean_now, y_mean_now)
    tls_err, _ = get_rmse_Loss_by_wb(x_test, y_test, tls_w, tls_b)
    # 训练
    # eta = [0.7, 0.8, 0.9, 0.8, 0.7]
    eta = [1.2, 1.1, 1.0, 1.1, 1.3]
    diag_x = np.eye(m)
    for i in range(m):
        diag_x[i][i] = eta[i]
    print(eta)
    w2 = tls2(np.dot(x_new, diag_x), y_new)  # x'=x*sigma_x  w'=sigma_x^(-1)*w  w=sigma_x*w'
    w_std2 = np.dot(diag_x, w2)  # m*m * m*1
    # 还原 w 计算 rmse
    tls_w2, tls_b2 = get_wb(w_std2, y_std_now / x_std_now, m, x_mean_now, y_mean_now)
    tls_err2, _ = get_rmse_Loss_by_wb(x_test, y_test, tls_w2, tls_b2)

    # 4）输出
    print("对角矩阵同: ", tls_err)
    print("随机设置异: ", tls_err2)

'''
diag_x: [0.7, 0.8, 0.9, 0.8, 0.7]
不加噪声
对角矩阵同:  0.07494489238991553
随机设置异:  0.07688782651295266
加噪声：
对角矩阵同:  0.07885943282835027
随机设置异:  0.08164049846471948


diag_x: [1.2, 1.1, 1.0, 1.1, 1.3]:
不加噪声。
对角矩阵同:  0.07494489238991553
随机设置异:  0.07412576174537679
添加随机噪声。
对角矩阵同:  0.07885943282835027
随机设置异:  0.07770453843911672
'''
