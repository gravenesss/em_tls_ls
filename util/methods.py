import numpy as np
from sklearn.linear_model import LinearRegression, ElasticNet, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from loss import getLossByWb_fn


# 求模型参数w, Y=WX+b (X^T X)^-1 X^T y。 直接使用 x_new, y_new 进行计算
def ls_fn(x_new, y_new):
    std_W = np.dot(np.dot(np.linalg.inv(np.dot(x_new.T, x_new)), x_new.T), y_new)
    return std_W


# 求模型参数w,  x y 在 tls内部未进行标准化  w未进行还原
def tls_fn(std_X, std_Y):
    # 定义矩阵B
    B = np.vstack((np.hstack((np.dot(std_X.T, std_X), np.dot(-std_X.T, std_Y))),
                   np.hstack((np.dot(-std_Y.T, std_X), np.dot(std_Y.T, std_Y)))))

    # 求B最小特征值对应的特征向量
    w, v = np.linalg.eigh(B)  # w特征值，v特征向量
    min_w_index = np.argsort(w)  # 最小特征值对应的下标，argsort(w)将w中的元素从小到大排列，输出其对应的下标
    min_w_v = v[:, min_w_index[0]].reshape(-1, 1)  # 最小特征值对应的特征向量

    # 求模型参数
    n = std_X.shape[1]  # 输入特征的个数
    std_W = (min_w_v[0:n] / min_w_v[n]).reshape(-1, 1)
    return std_W


# 通过标准化的w进行还原.
def getWb_fn(w_std, eta, m, x_mean, y_mean):
    # 求w
    w_std = w_std.reshape(1, -1)  # 一行
    w = eta * w_std  # 此处的乘法
    # print("还原：", w_std, w)

    # 求b
    tmp = 0
    for i in range(m):
        tmp += eta[i] * w_std[0][i] * x_mean[i]
    b = y_mean - tmp

    # 标量不需要变(w.ndim == 0), 之后需要转为列向量m*1
    w = w if w.ndim == 0 else w.reshape(-1, 1)
    return w, b


# 计算标准差，均值为0的情况，返回matrix每列的标准差 ok
def CalStd_fn(matrix):
    n, m = matrix.shape[0], 1 if matrix.ndim == 1 else matrix.shape[1]
    mean_col = 0  # 设置为0而不是原来的均值 np.mean(matrix, axis=0)
    diff_matrix = (matrix - mean_col) ** 2
    variance = np.sum(diff_matrix, axis=0) / n
    std = np.sqrt(variance)
    return std


# em 实验
def em_fn(x_now, y_now, m, x_std, y_std, x_mean, y_mean, x_test, y_test, w_epsilon=1e-6, correct=1e-2, convert_y='1'):
    # 假设样本误差E的每一列符合N(0,1) 和 标签误差r符合N(0,1)
    diag_x = np.eye(m)
    diag_x_inv = np.eye(m)
    flag = True
    w_pre = None
    # 记录 w 和 rmse
    wb_list = []
    rmse_list = []

    while flag:
        # 1.计算w
        w1 = tls_fn(np.dot(x_now, diag_x), y_now)  # x'=x*sigma_x  w'=sigma_x^(-1)*w  w=sigma_x*w'
        w_std = np.dot(diag_x, w1)  # m*m * m*1
        print("w_std.shape", w_std.shape, 'type:', type(w_std))
        # 还原 w 计算 rmse
        w_original, b_original = getWb_fn(w_std, y_std / x_std, m, x_mean, y_mean)
        rmse = getLossByWb_fn(x_test, y_test, w_original, b_original, err_type='rmse')
        wb_list.append(np.vstack((w_original, b_original)))
        rmse_list.append(rmse)

        # 2.根据 w、diag_x 计算 E 和 r
        w_t = np.transpose(w_std).reshape(1, -1)
        diag_x_inv2 = np.dot(diag_x_inv, diag_x_inv)
        denominator = np.dot(np.dot(w_t, diag_x_inv2), w_std) + 1  # wt: 1*m tmp_x:m*m  w:m*1
        r_up = (np.dot(x_now, w_std) - y_now).reshape(-1, 1)  # n*m * m*1 => n*1
        r = r_up / denominator  # 1*1
        E_up = -np.dot(np.dot(r_up, w_t), diag_x_inv2)  # n*1 * 1*m * m*m => n*m
        E = E_up / denominator
        print("E.shape:", E.shape, 'type:', type(E), "  r.shape:", r.shape, 'type:', type(r))

        # 3.更新sigma_x：根据样本误差的方差 和 标签误差的方差
        E_std = CalStd_fn(E)
        r_std = CalStd_fn(r)
        assert all(xi != 0.0 for xi in E_std), "样本误差 的标准差某一列存在为0的情况"  # assert expr, expr 为 False 时执行
        assert all(xi != 0.0 for xi in r_std), "标签误差 的标准差存在为0的情况"
        eta = (r_std + correct) / (E_std + correct)
        eta_inv = (E_std + correct) / (r_std + correct)
        for i in range(m):
            diag_x[i][i] = eta[i]
            diag_x_inv[i][i] = eta_inv[i]

        # 如果两次迭代的参数差距小于 w_epsilon 则结束循环
        if w_pre is None:
            w_pre = w_std
        else:
            gap = np.linalg.norm(w_std - w_pre)  # 欧氏距离
            w_pre = w_std
            flag = False if gap <= w_epsilon else True

    # plt.plot([x+1 for x in range(len(rmse_list))], rmse_list)
    # plt.show()

    print(rmse_list, '\nwb==== ==== ==== ==== ====\n', wb_list)
    sorted_data = sorted(zip(rmse_list, wb_list))  # 要根据 rmse_list 排序，需要记录
    print("sorted_data==== ==== ==== ==== ====\n", sorted_data)
    mid_rmse, mid_wb = sorted_data[len(sorted_data) // 2]
    mid_rmse = getLossByWb_fn(x_test, y_test, mid_wb[0:5], mid_wb[5], convert_y=convert_y)  # todo: 新加的 → 最后还原
    return mid_rmse, mid_wb


# 模型训练、进行了还原
def modelPredict_fn(x, y, x_test, y_test, model_name, convert_y='1', poly=2):
    if model_name == 'linear':
        # 线性回归
        model = LinearRegression()
    elif model_name == 'tree':
        # 决策树回归
        model = DecisionTreeRegressor()
    elif model_name == 'svr':
        # 支持向量回归
        model = SVR()
    elif model_name == 'rf':
        # 随机森林回归
        model = RandomForestRegressor()
    elif model_name == 'knn':
        # K最近邻回归
        model = KNeighborsRegressor()
    elif model_name == 'poly':
        # 多项式回归: PolynomialFeatures(2) 是一个用于生成二次多项式特征的变换器
        model = make_pipeline(PolynomialFeatures(poly), LinearRegression())
        pass
    elif model_name == 'en':
        # 弹性网络回归
        model = ElasticNet()
    elif model_name == 'lasso':
        # 套索回归
        model = Lasso(alpha=0.01)
    else:  # 默认使用决策树
        model = DecisionTreeRegressor()

    # 如果你的 y 是二维的，可以使用 y.ravel() 将其转换为一维数组。
    # 如果你有一个列向量 y，可以使用 np.squeeze(y) 或 y.flatten() 来调整其形状。
    model.fit(x, y.flatten())
    y_predict = model.predict(x_test)

    if convert_y == 'log10':  # log10(y) = y' y=10^y'
        y_predict = np.power(10, y_predict)
        y_test = np.power(10, y_test)
    elif convert_y == 'loge':
        y_predict = np.exp(y_predict)
        y_test = np.exp(y_test)
    else:
        pass

    rmse = np.sqrt(mean_squared_error(y_test, y_predict))
    return rmse
    pass
