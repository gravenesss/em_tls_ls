import copy
from scipy.spatial import KDTree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression, ElasticNet, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
# from util.loss import getLossByWb_fn

np.set_printoptions(linewidth=np.inf)  # 设置ndaary一行显示，不折行


# 求模型参数w, Y=WX+b (X^T X)^-1 X^T y。 直接使用 x_new, y_new 进行计算
def ls_fn(x_new, y_new):
    std_w = np.linalg.inv(x_new.T.dot(x_new)).dot(x_new.T).dot(y_new)
    return std_w


# 求模型参数w,  x y 在 tls内部未进行标准化  w未进行还原
def tls_fn(std_x, std_y):
    # 定义矩阵B
    B = np.vstack((np.hstack((np.dot(std_x.T, std_x), np.dot(-std_x.T, std_y))),
                   np.hstack((np.dot(-std_y.T, std_x), np.dot(std_y.T, std_y)))))
    # print("B==== ==== ==== ==== \n", B)

    # 求B最小特征值对应的特征向量
    w, v = np.linalg.eigh(B)  # w特征值，v特征向量
    min_w_index = np.argsort(w)  # 最小特征值对应的下标，argsort(w)将w中的元素从小到大排列，输出其对应的下标
    min_w_v = v[:, min_w_index[0]].reshape(-1, 1)  # 最小特征值对应的特征向量
    # print("特征值，特征向量：", w, v)
    # print("min_value : ", w[min_w_index][0])
    # print("min_vector: ", min_w_v)

    # 求模型参数
    m = std_x.shape[1]  # 输入特征的个数 m*1
    std_w = (min_w_v[0:m] / min_w_v[m]).reshape(-1, 1)
    # print("w==== ==== ==== ==== \n", std_w)
    return std_w


def tls2(x_std, y_std):
    # 构造矩阵 B
    XTX = x_std.T @ x_std
    XTY = x_std.T @ y_std
    YTX = y_std.T @ x_std
    YTY = np.sum(y_std**2)
    B = np.block([[XTX, -XTY], [-YTX, YTY]])

    # 计算B的特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eig(B)

    # 找到最小特征值对应的特征向量  np.argmin()在给定数组中找到最小值的索引
    min_eigenvalue_index = np.argmin(eigenvalues)
    z = eigenvectors[:, min_eigenvalue_index].reshape(-1, 1)

    # 计算 w_tls
    w_tls = z[:-1] / z[-1].reshape(-1, 1)

    return w_tls


# 获取原先wb
def getWb_fn(n, std_w, std_x, mean_x, std_y, mean_y):
    # std_w, standard_x, mean_x： m*1； standard_y, mean_y： 1*1.
    original_w = (std_w @ std_y / std_x).reshape(-1, 1)  # (m,1) / (m,1) 对应位置进行除法
    tmp = 0
    for i in range(n):
        tmp = tmp + std_w[i] * mean_x[i] * std_y / std_x[i]
    original_b = mean_y - tmp
    return original_w, original_b


# 计算标准差，均值为0的情况，返回matrix每列的标准差 ok
def calStd_fn(matrix):
    n, m = matrix.shape[0], 1 if matrix.ndim == 1 else matrix.shape[1]
    mean_col = 0  # 设置为0而不是原来的均值 np.mean(matrix, axis=0)
    diff_matrix = (matrix - mean_col) ** 2
    variance = np.sum(diff_matrix, axis=0) / n
    std = np.sqrt(variance)
    return std

def compute_E2r2(X1, E1, r1, X2, distance='kdt'):
    E2 = np.zeros((X2.shape[0], X1.shape[1]))
    r2 = np.zeros((X2.shape[0], 1))
    tree = KDTree(X1)

    for i, x2_row in enumerate(X2):
        if distance == 'l2':
            # 计算X2中每一行与X1中所有行的距离
            distances = np.linalg.norm(X1 - x2_row, axis=1)
            index = np.argmin(distances)
        else:  # kd树
            dist, index = tree.query(x2_row)

        E2[i] = E1[index]
        r2[i] = r1[index]

    return E2, r2

#
def em_fn(xx_train, yy_train, xx_test, yy_test, xx_now, yy_now, xx_std, yy_std, xx_mean, yy_mean,
          w_epsilon=1e-6, correct=1e-2, convert_y='1'):
    flag = True
    m = xx_now.shape[1]
    # 假设样本误差E的每一列符合N(0,1) 和 标签误差r符合N(0,1)
    diag_x = np.eye(m)
    diag_x_inv = np.eye(m)
    w_std = tls_fn(xx_now, yy_now)
    w_pre = w_std

    test_x_std = scale(xx_test)

    # 记录 w 和 rmse
    wb_list = []
    train1_list = []
    train2_list = []
    train3_list = []
    train4_list = []
    test1_list = []
    test2_list = []
    test3_list = []
    test4_list = []
    target_list = []

    train_x = copy.deepcopy(xx_train)
    train_y = copy.deepcopy(yy_train)
    test_x = copy.deepcopy(xx_test)
    test_y = copy.deepcopy(yy_test)
    now_x = copy.deepcopy(xx_now)
    now_y = copy.deepcopy(yy_now)
    std_x = copy.deepcopy(xx_std)
    std_y = copy.deepcopy(yy_std)
    mean_x = copy.deepcopy(xx_mean)
    mean_y = copy.deepcopy(yy_mean)

    while flag:
        # E步：
        # 1.1: 计算 r E
        wT = np.transpose(w_std).reshape(1, -1)  # 1*m
        diag_x_inv2 = diag_x_inv @ diag_x_inv  # m*m
        denominator = wT @ diag_x_inv2 @ w_std + 1  # wt: 1*m tmp_x:m*m  w:m*1 → 1*1
        r = ((now_x @ w_std - now_y) / denominator).reshape(-1, 1)  # n*m * m*1 => n*1 n=124*0.9=111
        E = -r @ wT @ diag_x_inv2  # n*1 * 1*m * m*m => n*m 111*5
        # print("E.shape:", E.shape, "  r.shape:", r.shape)
        # 1.2: 更新 diag_x
        E_std = calStd_fn(E)
        r_std = calStd_fn(r)
        # print("E_std:", E_std, "r_std:", r_std)
        assert all(xi != 0.0 for xi in E_std), "样本误差 的标准差某一列存在为0的情况"  # assert expr, expr 为 False 时执行
        assert all(xi != 0.0 for xi in r_std), "标签误差 的标准差存在为0的情况"
        for i in range(m):
            diag_x[i][i] = (r_std + correct) / (E_std[i] + correct)
            diag_x_inv[i][i] = (E_std[i] + correct) / (r_std + correct)
        # print("diag_x.shape:", diag_x.shape, "\n", diag_x, "\ndiag_x_inv.shape:", diag_x_inv.shape, "\n", diag_x_inv)

        # 计算 target
        r_norm = np.sum(r ** 2)  # L2范数的平方 sqrt 再平方。
        E_norm = np.linalg.norm(E @ diag_x, 'fro') ** 2  # F 范数的平方
        lambda_r = 2 * np.transpose(r)  # 1*n
        lapse = (now_x + E) @ w_std - now_y - r  # n*1
        target = E_norm + r_norm + lambda_r @ lapse
        target_list.append(target[0][0])
        # print("lapse:", np.sqrt(np.sum(lapse ** 2)))
        # print('target:', target)

        # M步: 计算 w_std
        w1 = tls_fn(now_x @ diag_x, now_y)  # x'=x*diag_x  w'=diag_x_inv*w  w=diag_x*w'  → w1 m*1
        w_std = diag_x @ w1  # m*m * m*1 = m*1 m=5
        w_original, b_original = getWb_fn(m, w_std, std_x, mean_x, std_y, mean_y)
        wb_list.append(np.vstack((w_original, b_original)))
        # print("w_std.shape", w_std.shape)

        # 训练误差：
        # print("rmse:========================")
        train_rmse1 = np.sqrt(mean_squared_error(train_y, train_x @ w_original + b_original))  # wb还原
        train_rmse2 = np.sqrt(mean_squared_error(train_y, now_x @ w_std * std_y + mean_y))  # mean-std还原
        train_rmse3 = np.sqrt(mean_squared_error(train_y, (train_x + E) @ w_original - r + b_original))  # train_Er
        train_rmse4 = np.sqrt(mean_squared_error(train_y, ((now_x + E) @ w_std - r) * std_y + mean_y))  # std_Er

        # 测试误差: rmse_test = getLossByWb_fn(x_test, y_test, w_original, b_original, err_type='rmse', convert_y=convert_y)
        E2, r2 = compute_E2r2(train_x, E, r, test_x)
        test_rmse1 = np.sqrt(mean_squared_error(test_y, test_x @ w_original + b_original))
        test_rmse2 = np.sqrt(mean_squared_error(test_y, test_x_std @ w_std * std_y + mean_y))
        test_rmse3 = np.sqrt(mean_squared_error(test_y, (test_x + E2) @ w_original - r2 + b_original))  # train_Er
        test_rmse4 = np.sqrt(mean_squared_error(test_y, ((test_x_std + E2) @ w_std - r2) * std_y + mean_y))  # std_Er

        # 记录到数组
        train1_list.append(train_rmse1)
        train2_list.append(train_rmse2)
        train3_list.append(train_rmse3)
        train4_list.append(train_rmse4)
        test1_list.append(test_rmse1)
        test2_list.append(test_rmse2)
        test3_list.append(test_rmse3)
        test4_list.append(test_rmse4)

        # 判断是否结束循环
        gap = np.linalg.norm(w_std - w_pre)  # 欧氏距离
        w_pre = w_std
        flag = False if gap <= w_epsilon else True

    if True:
        fig, axs = plt.subplots(3, 4, figsize=(15, 10))
        ys = [train1_list, train2_list, train3_list, train4_list, test1_list, test2_list, test3_list, test4_list, target_list]
        labels = ['train_wb', 'train_ms', 'train_Er', 'train_std_Er', 'test_wb', 'test_ms', 'test_Er', 'test_std_Er', 'target']
        for i in range(len(ys)):  # for i, ax in enumerate(axs.flat):
            axs[i // 4, i % 4].plot(ys[i], label=labels[i])
            axs[i // 4, i % 4].set_title(labels[i])
            axs[i // 4, i % 4].legend()

        plt.show()
    # print("rmse==== ==== ==== ==== ====\n", rmse_list, '\nwb==== ==== ==== ==== ====\n', wb_list)

    sorted_data = sorted(zip(test1_list, wb_list))  # 要根据 rmse_list 排序，需要记录
    mid_rmse, mid_wb = sorted_data[len(sorted_data) // 2]
    # return mid_rmse, mid_wb
    return test1_list[-1], wb_list[-1]


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


# 标准化后的训练集；测试集；标准化前的均值，标准差。
# x_new, y_new, x_test, y_test, x_mean_now, y_mean_now, x_std_now, y_std_now = dataProcess_fn(0.1, True)
def dataProcess1_fn(data_x, data_y, noise_pattern, test_ratio, add_noise, now_seed=42):
    # 1）划分数据集
    x, x_test, y, y_test = train_test_split(data_x, data_y, test_size=test_ratio, random_state=now_seed)
    if add_noise:
        print("添加随机噪声。")
        noise_ratio = 0.2
        x_ratio = noise_ratio * noise_pattern[:-1]  # b.噪声比例 和 c.每一列权重
        y_ratio = noise_ratio * noise_pattern[-1]
        x_std_pre = np.std(x, axis=0)  # d. x y 的标准差
        y_std_pre = np.std(y, axis=0)
        new_std_x = np.multiply(x_std_pre, x_ratio)
        new_std_y = np.multiply(y_std_pre, y_ratio)
        np.random.seed(now_seed)  # a. 随机种子
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

    return x, y, x_new, y_new, x_test, y_test, x_mean_now, y_mean_now, x_std_now, y_std_now


# 先标准化，再加噪声
def dataProcess_fn(data_x, data_y, noise_pattern, test_ratio, add_noise, now_seed=42):
    x, x_test, y, y_test = train_test_split(data_x, data_y, test_size=test_ratio, random_state=now_seed)
    # 1）进行标准化
    x_mean_now = np.mean(x, axis=0)
    y_mean_now = np.mean(y, axis=0)
    x_std_now = np.std(x, axis=0)  # em中使用之后的 std
    y_std_now = np.std(y, axis=0)
    x_now = scale(x)
    y_now = scale(y)

    if add_noise:
        print("添加随机噪声。")
        noise_ratio = 0.2
        x_ratio = noise_ratio * noise_pattern[:-1]  # b.噪声比例 和 c.每一列权重
        y_ratio = noise_ratio * noise_pattern[-1]
        # x_std_pre = np.std(x, axis=0)  # d. x y 的标准差  现在是全1
        # y_std_pre = np.std(y, axis=0)
        # new_std_x = np.multiply(x_std_pre, x_ratio)
        # new_std_y = np.multiply(y_std_pre, y_ratio)
        np.random.seed(now_seed)  # a. 随机种子
        x_new = x_now + np.random.normal(0, x_ratio, x.shape)
        y_new = y_now + np.random.normal(0, y_ratio, y.shape)
    else:
        print("不加噪声。")
        x_new = x_now
        y_new = y_now

    return x, y, x_new, y_new, x_test, y_test, x_mean_now, y_mean_now, x_std_now, y_std_now
