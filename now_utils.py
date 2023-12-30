from datetime import datetime

from scipy.spatial import KDTree
from sklearn.preprocessing import scale, StandardScaler
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, ElasticNet, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from util.methods import tls_fn
from util.plot_result import plotXWbs_fn

np.set_printoptions(linewidth=np.inf)  # 设置ndaary一行显示，不折行


def rmse(y_true, y_pred):
    return np.sqrt(sum(np.square(y_true - y_pred)) / len(y_true))


def std_xy(x_train, y_train):
    stand_scaler = StandardScaler()
    # 标准化X_train
    standard_X = np.std(x_train, axis=0).reshape(-1, 1)
    std_X = stand_scaler.fit_transform(x_train)
    mean_X = stand_scaler.mean_.reshape(-1, 1)

    # 标准化Y_train
    standard_Y = np.std(y_train, axis=0).reshape(-1, 1)
    mean_Y = np.array(np.mean(y_train)).reshape(-1, 1)
    std_Y = (y_train - mean_Y) / standard_Y
    return standard_X, std_X, mean_X, standard_Y, std_Y, mean_Y
    pass


def get_wb(n, std_w, standard_x, mean_x, standard_y, mean_y):
    W = np.dot(std_w, standard_y) / standard_x
    _ = 0
    for i in range(n):
        _ = _ + std_w[i] * mean_x[i] * standard_y / standard_x[i]
    b = mean_y - _
    return W, b
    pass


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


def em_fn(train_x, train_y, w_epsilon=1e-6, now_correct=1e-2, max_iteration=24):
    std_x = np.std(train_x, axis=0)
    mean_x = np.mean(train_x, axis=0)
    std_y = np.std(train_y, axis=0)
    mean_y = np.mean(train_y, axis=0)
    now_x = scale(train_x)
    now_y = scale(train_y)

    flag = True
    m = now_x.shape[1]
    diag_x = np.eye(m)
    diag_x_inv = np.eye(m)
    w_std = tls_fn(now_x, now_y)
    w_pre = w_std

    # 记录 w, b, E, r
    w_original, b_original, E, r = None, None, None, None
    iteration = 0
    while flag and iteration < max_iteration:
        # E步-1.1: 计算 r E
        wT = w_std.T.reshape(1, -1)  # 1*m
        diag_x_inv2 = diag_x_inv @ diag_x_inv  # m*m
        denominator = wT @ diag_x_inv2 @ w_std + 1  # wt: 1*m tmp_x:m*m  w:m*1 → 1*1
        r = ((now_x @ w_std - now_y) / denominator).reshape(-1, 1)  # n*m * m*1 => n*1 n=124*0.9=111
        E = -r @ wT @ diag_x_inv2  # n*1 * 1*m * m*m => n*m 111*5
        # E步-1.2: 更新 diag_x
        E_std = calStd_fn(E)
        r_std = calStd_fn(r)
        for i in range(m):
            diag_x[i][i] = (r_std + now_correct) / (E_std[i] + now_correct)
            diag_x_inv[i][i] = (E_std[i] + now_correct) / (r_std + now_correct)

        # M步: 计算 w_std
        w1 = tls_fn(now_x @ diag_x, now_y)  # x'=x*diag_x  w'=diag_x_inv*w  w=diag_x*w'  → w1 m*1
        w_std = diag_x @ w1  # m*m * m*1 = m*1 m=5

        # 判断是否结束循环
        gap = np.linalg.norm(w_std - w_pre)  # 欧氏距离
        w_pre = w_std
        flag = False if gap <= w_epsilon else True
        iteration += 1

    w_original, b_original = getWb_fn(m, w_std, std_x, mean_x, std_y, mean_y)

    return w_original, b_original, E, r


def emTest_fn(train_x, train_y, w_epsilon=1e-6, now_correct=1e-2, max_iteration=24, plot_stds=False):
    std_x = np.std(train_x, axis=0)
    mean_x = np.mean(train_x, axis=0)
    std_y = np.std(train_y, axis=0)
    mean_y = np.mean(train_y, axis=0)
    now_x = StandardScaler().fit_transform(train_x)
    now_y = StandardScaler().fit_transform(train_y)

    flag = True
    m = now_x.shape[1]
    diag_x = np.eye(m)
    diag_x_inv = np.eye(m)
    w_std = tls_fn(now_x, now_y)
    w_pre = w_std

    feature_lapse_stds = np.zeros(m, dtype=object)
    feature_lapse_stds[:] = [[] for _ in range(m)]
    life_lapse_stds = []
    r_div_E_stds = np.zeros(m, dtype=object)
    r_div_E_stds[:] = [[] for _ in range(m)]

    # 记录 w, b, E, r
    target_list1, target_list2, target_list3 = [], [], []
    w_list, b_list = [], []
    w_std_list, wb_list = [], []
    w_original, b_original, E, r = None, None, None, None
    iteration = 0
    while flag and iteration < max_iteration:
        # E步-1.1: 计算 r E
        wT = w_std.T.reshape(1, -1)  # 1*m
        diag_x_inv2 = diag_x_inv @ diag_x_inv  # m*m
        denominator = wT @ diag_x_inv2 @ w_std + 1  # wt: 1*m tmp_x:m*m  w:m*1 → 1*1
        r = ((now_x @ w_std - now_y) / denominator).reshape(-1, 1)  # n*m * m*1 => n*1 n=124*0.9=111
        E = -r @ wT @ diag_x_inv2  # n*1 * 1*m * m*m => n*m 111*5
        # E步-1.2: 更新 diag_x
        E_std = np.std(E, axis=0)  # calStd_fn(E)
        r_std = np.std(r)   # calStd_fn(r)
        for j in range(m):
            diag_x[j][j] = (r_std + now_correct) / (E_std[j] + now_correct)
            diag_x_inv[j][j] = (E_std[j] + now_correct) / (r_std + now_correct)

        # 正式训练不需要：查看eta的情况
        # assert all(xi != 0.0 for xi in E_std), "样本误差 的标准差某一列存在为0的情况"  # assert expr, expr 为 False 时执行
        # assert all(xi != 0.0 for xi in r_std), "标签误差 的标准差存在为0的情况"
        eta = (r_std + now_correct) / (E_std + now_correct)
        # print("r_std:", r_std, "E_std:", E_std)
        # print("eta:", eta)
        life_lapse_stds.append(r_std)
        for i in range(m):
            feature_lapse_stds[i].append(E_std[i])
            r_div_E_stds[i].append(eta[i])
        # print("E.shape:", E.shape, "  r.shape:", r.shape)
        # print("diag_x:", diag_x)
        # print("diag_x_inv", diag_x_inv)

        # 正式训练不需要：计算 target
        r_norm = np.sum(r ** 2)  # L2范数的平方
        E_norm = np.linalg.norm(E @ diag_x, 'fro') ** 2  # F 范数的平方
        lambda_r = 2 * r.T  # 1*n
        lapse = (now_x + E) @ w_std - now_y - r  # n*1
        target = E_norm + r_norm + lambda_r @ lapse
        target_list1.append(target[0][0])
        # print("lapse:", lambda_r @ lapse)
        # print('target1:', target[0][0])

        # 正式训练不需要：(now_x + E) @ w_std - now_y - r
        loss = ((now_x + E) @ w_std - now_y - r)[0]
        target_list2.append(loss)

        # M步: 计算 w_std
        w1 = tls_fn(now_x @ diag_x, now_y)  # x'=x*diag_x  w'=diag_x_inv*w  w=diag_x*w'  → w1 m*1
        w_std = diag_x @ w1  # m*m * m*1 = m*1 m=5
        w_std_list.append(w_std.flatten().tolist())

        # 正式训练不需要：计算 w_original b_original 返回
        w_original, b_original = getWb_fn(m, w_std, std_x, mean_x, std_y, mean_y)
        w_list.append(w_original)
        b_list.append(b_original)
        wb_list.append(np.vstack((w_original, b_original)).flatten().tolist())
        # print("w_original, b_original:", w_original, b_original)

        # 判断是否结束循环
        gap = np.linalg.norm(w_std - w_pre)  # 欧氏距离
        w_pre = w_std
        flag = False if gap <= w_epsilon else True
        iteration += 1

    # w_original, b_original = getWb_fn(m, w_std, std_x, mean_x, std_y, mean_y)

    if plot_stds:
        plt.plot(life_lapse_stds, label='r_std')
        plt.legend()
        fig0, axs0 = plt.subplots(2, m, figsize=(12, 8))
        for i in range(m):
            axs0[0, i].plot(feature_lapse_stds[i], label='E_std'+str(i))
            axs0[1, i].plot(r_div_E_stds[i], label='r_div_E_stds'+str(i))
            axs0[0, i].legend()
            axs0[1, i].legend()

        # 绘制 w_std 的图像：
        w_std_list = np.array(w_std_list)
        fig1, axs1 = plt.subplots(1, m, figsize=(10, 3))
        for i in range(m):
            axs1[i].plot(w_std_list[:, i], label='w_std'+str(i), marker='*')
            axs1[i].legend()

        wb_list = np.array(wb_list)
        cur_dir = 'em_test'
        file_name = datetime.now().strftime("%Y%m%d%H%M%S") + ' w_b.png'
        plotXWbs_fn([ii for ii in range(1, 1+len(wb_list))], [wb_list], 'iterator', ['wb_original'],
                    ['*'], m, cur_dir, file_name, need_show=True, need_save=False)

        plt.show()
        plt.close()

    return w_list, b_list, E, r, target_list1, target_list2


def emTest_fn1(train_x, train_y, test_x, test_y, seed, w_epsilon=1e-6, now_correct=1e-2, plot_pic=False):
    std_x = np.std(train_x, axis=0)
    mean_x = np.mean(train_x, axis=0)
    std_y = np.std(train_y, axis=0)
    mean_y = np.mean(train_y, axis=0)
    now_x = scale(train_x)
    now_y = scale(train_y)
    test_x_std = scale(test_x)

    flag = True
    m = now_x.shape[1]
    diag_x = np.eye(m)
    diag_x_inv = np.eye(m)
    w_std = tls_fn(now_x, now_y)
    w_pre = w_std

    # 记录 w 和 rmse
    train1_list = []
    train2_list = []
    train3_list = []
    train4_list = []
    test1_list = []
    test2_list = []
    test3_list = []
    test4_list = []
    target_list = []

    iteration = 0
    max_iteration = 64
    while flag and iteration < max_iteration:
        # E步-1.1: 计算 r E
        wT = w_std.T.reshape(1, -1)  # 1*m
        diag_x_inv2 = diag_x_inv @ diag_x_inv  # m*m
        denominator = wT @ diag_x_inv2 @ w_std + 1  # wt: 1*m tmp_x:m*m  w:m*1 → 1*1
        r = ((now_x @ w_std - now_y) / denominator).reshape(-1, 1)  # n*m * m*1 => n*1 n=124*0.9=111
        E = -r @ wT @ diag_x_inv2  # n*1 * 1*m * m*m => n*m 111*5
        # E步-1.2: 更新 diag_x
        E_std = calStd_fn(E)
        r_std = calStd_fn(r)
        for i in range(m):
            diag_x[i][i] = (r_std + now_correct) / (E_std[i] + now_correct)
            diag_x_inv[i][i] = (E_std[i] + now_correct) / (r_std + now_correct)

        assert all(xi != 0.0 for xi in E_std), "样本误差 的标准差某一列存在为0的情况"  # assert expr, expr 为 False 时执行
        assert all(xi != 0.0 for xi in r_std), "标签误差 的标准差存在为0的情况"
        print("eta:", (r_std + now_correct) / (E_std + now_correct))
        # print("E.shape:", E.shape, "  r.shape:", r.shape)
        # print("E_std:", E_std, "r_std:", r_std)
        # print("diag_x:", diag_x)
        # print("diag_x_inv", diag_x_inv)

        # 计算 target，正式训练不需要
        r_norm = np.sum(r ** 2)  # L2范数的平方 sqrt 再平方。
        E_norm = np.linalg.norm(E @ diag_x, 'fro') ** 2  # F 范数的平方
        lambda_r = 2 * r.T  # 1*n
        lapse = (now_x + E) @ w_std - now_y - r  # n*1
        target = E_norm + r_norm + lambda_r @ lapse
        target_list.append(target[0][0])
        # print("lapse:", lambda_r @ lapse)
        print('target:', target)

        # M步: 计算 w_std
        w1 = tls_fn(now_x @ diag_x, now_y)  # x'=x*diag_x  w'=diag_x_inv*w  w=diag_x*w'  → w1 m*1
        w_std = diag_x @ w1  # m*m * m*1 = m*1 m=5
        w_original, b_original = getWb_fn(m, w_std, std_x, mean_x, std_y, mean_y)
        # print("w_std.shape", w_std.shape)

        # 训练误差： 正式训练不需要，放在外面计算。
        train_rmse1 = np.sqrt(mean_squared_error(train_y, train_x @ w_original + b_original))  # wb还原
        train_rmse2 = np.sqrt(mean_squared_error(train_y, now_x @ w_std * std_y + mean_y))  # mean-std还原
        train_rmse3 = np.sqrt(mean_squared_error(train_y, (train_x + E) @ w_original - r + b_original))  # train_Er
        train_rmse4 = np.sqrt(mean_squared_error(train_y, ((now_x + E) @ w_std - r) * std_y + mean_y))  # std_Er
        # 测试误差: rmse_test = getLossByWb_fn(x_test, y_test, w_original, b_original, err_type='rmse', convert_y=convert_y)
        E2, r2 = compute_E2r2(train_x, E, r, test_x, 'l2')
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
        iteration += 1

    len_em = len(train1_list)
    tls_w_std = tls_fn(now_x, now_y)
    tls_w, tls_b = getWb_fn(m, tls_w_std, std_x, mean_x, std_y, mean_y)
    tls_train1 = [np.sqrt(mean_squared_error(train_y, train_x @ tls_w + tls_b))] * len_em
    tls_train2 = [np.sqrt(mean_squared_error(train_y, now_x @ tls_w_std * std_y + mean_y))] * len_em
    tls_test1 = [np.sqrt(mean_squared_error(test_y, test_x @ tls_w + tls_b))] * len_em
    tls_test2 = [np.sqrt(mean_squared_error(test_y, test_x_std @ tls_w_std * std_y + mean_y))] * len_em

    if plot_pic:
        fig, axs = plt.subplots(3, 4, figsize=(15, 10))
        tlss = [tls_train1, tls_train2, tls_train1, tls_train2, tls_test1, tls_test2, tls_test1, tls_test2, []]
        ems = [train1_list, train2_list, train3_list, train4_list, test1_list, test2_list, test3_list, test4_list,
               target_list]
        labels = ['train_wb', 'train_ms', 'train_Er', 'train_std_Er', 'test_wb', 'test_ms', 'test_Er', 'test_std_Er',
                  'target']
        for i in range(len(ems)):  # for i, ax in enumerate(axs.flat):
            axs[i // 4, i % 4].plot(tlss[i], label='tls')
            axs[i // 4, i % 4].plot(ems[i], label='em')
            axs[i // 4, i % 4].set_title(labels[i])
            axs[i // 4, i % 4].legend()

        fig.suptitle('tls vs em with seed ' + str(seed), fontsize=16)
        plt.show()
    print("tls-err: ", tls_test1[-1], "\nem-err: ", test1_list[-1])
    return test1_list[-1], tls_test1[-1]


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
