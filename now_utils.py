from datetime import datetime
from sklearn.preprocessing import scale, StandardScaler
import numpy as np
from matplotlib import pyplot as plt
from util.plot_result import plotXWbs_fn

np.set_printoptions(linewidth=np.inf)  # 设置ndaary一行显示，不折行


def rmse(y_true, y_pred):
    return np.sqrt(sum(np.square(y_true - y_pred)) / len(y_true))


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


def tls(x_train, y_train):
    # 标准化X_train
    standard_X = np.std(x_train, axis=0).reshape(-1, 1)
    stand_scaler = StandardScaler()
    std_X = stand_scaler.fit_transform(x_train)
    mean_X = stand_scaler.mean_.reshape(-1, 1)

    # 标准化Y_train
    mean_Y = np.array(np.mean(y_train)).reshape(-1, 1)
    standard_Y = np.std(y_train, axis=0).reshape(-1, 1)
    std_Y = (y_train - mean_Y) / standard_Y

    # 定义矩阵B
    B = np.vstack((np.hstack((np.dot(std_X.T, std_X), np.dot(-std_X.T, std_Y))),
                   np.hstack((np.dot(-std_Y.T, std_X), np.dot(std_Y.T, std_Y)))))

    # 求B最小特征值对应的特征向量
    w, v = np.linalg.eigh(B)  # w特征值，v特征向量
    min_w_index = np.argsort(w)  # 最小特征值对应的下标，argsort(w)将w中的元素从小到大排列，输出其对应的下标
    min_w_v = v[:, min_w_index[0]].reshape(-1, 1)  # 最小特征值对应的特征向量
    # min_w_v = v[min_w_index[0], :].reshape(-1, 1)
    n = std_X.shape[1]  # 输入特征的个数
    std_W = (min_w_v[0:n] / min_w_v[n]).reshape(-1, 1)

    # 求模型参数
    W = np.dot(std_W, standard_Y) / standard_X
    # 计算b
    _ = 0
    for i in range(n):
        _ = _ + std_W[i] * mean_X[i] * standard_Y / standard_X[i]
    b = mean_Y - _

    return W, b


def ls(x_train, y_train):
    # 标准化X_train
    standard_X = np.std(x_train, axis=0).reshape(-1, 1)  # 加入噪声后的标准差
    stand_scaler = StandardScaler()
    std_X = stand_scaler.fit_transform(x_train)
    mean_X = stand_scaler.mean_.reshape(-1, 1)

    # 标准化Y_train
    mean_Y = np.array(np.mean(y_train)).reshape(-1, 1)
    standard_Y = np.std(y_train, axis=0).reshape(-1, 1)
    std_Y = (y_train - mean_Y) / standard_Y

    # 求模型参数,Y=WX+b
    std_W = np.dot(np.dot(np.linalg.inv(np.dot(std_X.T, std_X)), std_X.T), std_Y)
    W = np.dot(std_W, standard_Y) / standard_X

    # 计算b
    n = x_train.shape[1]
    _ = 0
    for i in range(n):
        _ = _ + std_W[i] * mean_X[i] * standard_Y / standard_X[i]
    b = mean_Y - _
    return W, b


# 求模型参数w, Y=WX+b (X^T X)^-1 X^T y。 直接使用 x_new, y_new 进行计算
def ls_std_fn(x_new, y_new):
    std_w = np.linalg.inv(x_new.T.dot(x_new)).dot(x_new.T).dot(y_new)
    return std_w


# 求模型参数w,  x y 在 tls内部未进行标准化  w未进行还原
def tls_std_fn(std_x, std_y):
    # 定义矩阵B
    B = np.vstack((np.hstack((std_x.T @ std_x, -std_x.T @ std_y)),
                   np.hstack((-std_y.T @ std_x, std_y.T @ std_y))
                   ))
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
    w1 = tls_std_fn(now_x @ diag_x, now_y)
    w_std = w_pre = diag_x @ w1

    iteration = 0
    while flag and iteration < max_iteration:
        # E步-1.1: 计算 r E
        wT = w_std.T.reshape(1, -1)  # 1*m
        diag_x_inv2 = diag_x_inv @ diag_x_inv  # m*m
        denominator = 1 + wT @ diag_x_inv2 @ w_std  # wt: 1*m tmp_x:m*m  w:m*1 → 1*1
        r = ((now_x @ w_std - now_y) / denominator).reshape(-1, 1)  # n*m * m*1 => n*1 n=124*0.9=111
        E = -r @ wT @ diag_x_inv2  # n*1 * 1*m * m*m => n*m 111*5
        # E步-1.2: 更新 diag_x
        E_std = calStd_fn(E)
        r_std = calStd_fn(r)
        for j in range(m):
            diag_x[j][j] = (r_std + now_correct) / (E_std[j] + now_correct)
            diag_x_inv[j][j] = (E_std[j] + now_correct) / (r_std + now_correct)

        # M步: 计算 w_std
        w1 = tls_std_fn(now_x @ diag_x, now_y)  # x'=x*diag_x  w'=diag_x_inv*w  w=diag_x*w'  → w1 m*1
        w_std = diag_x @ w1  # m*m * m*1 = m*1 m=5

        # 判断是否结束循环
        gap = np.linalg.norm(w_std - w_pre)
        w_pre = w_std
        flag = False if gap <= w_epsilon else True
        iteration += 1

    w_original, b_original = getWb_fn(m, w_std, std_x, mean_x, std_y, mean_y)
    return w_original, b_original


def emTest_fn(train_x, train_y, w_epsilon=1e-6, now_correct=1e-2, max_iteration=24, plot_stds_wb=False):
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
    w1 = tls_std_fn(now_x @ diag_x, now_y)
    w_std = w_pre = diag_x @ w1

    # E_stds
    feature_lapse_stds = np.zeros(m, dtype=object)
    feature_lapse_stds[:] = [[] for _ in range(m)]
    # r_stds
    life_lapse_stds = []
    # diag_x
    r_div_E_stds = np.zeros(m, dtype=object)
    # diag_x_inv
    r_div_E_stds[:] = [[] for _ in range(m)]
    # 记录目标值, w_list, b_list, w_std_list, wb_list
    target_list1, target_list2, target_list3 = [], [], []
    w_list, b_list = [], []
    w_std_list, wb_list = [], []

    # 记录 w, b, E, r
    w_original, b_original, E, r = None, None, None, None
    iteration = 0
    while flag and iteration < max_iteration:
        # E步-1.1: 计算 r E
        wT = w_std.T.reshape(1, -1)  # 1*m
        diag_x_inv2 = diag_x_inv @ diag_x_inv  # m*m
        denominator = 1 + wT @ diag_x_inv2 @ w_std  # wt: 1*m tmp_x:m*m  w:m*1 → 1*1
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
        w1 = tls_std_fn(now_x @ diag_x, now_y)  # x'=x*diag_x  w'=diag_x_inv*w  w=diag_x*w'  → w1 m*1
        w_std = diag_x @ w1  # m*m * m*1 = m*1 m=5
        w_std_list.append(w_std.flatten().tolist())

        # 正式训练不需要：计算 w_original b_original 返回
        w_original, b_original = getWb_fn(m, w_std, std_x, mean_x, std_y, mean_y)
        w_list.append(w_original)
        b_list.append(b_original)
        wb_list.append(np.vstack((w_original, b_original)).flatten().tolist())
        # print("w_original, b_original:", w_original, b_original)

        # 判断是否结束循环
        gap = np.linalg.norm(w_std - w_pre)
        w_pre = w_std
        flag = False if gap <= w_epsilon else True
        iteration += 1

    # w_original, b_original = getWb_fn(m, w_std, std_x, mean_x, std_y, mean_y)

    if plot_stds_wb:
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
        cur_dir = 'test_em'
        file_name = datetime.now().strftime("%Y%m%d%H%M%S") + ' w_b.png'
        plotXWbs_fn([ii for ii in range(1, 1+len(wb_list))], [wb_list], 'iterator', ['wb_original'],
                    ['*'], m, cur_dir, file_name, need_show=True, need_save=False)

        plt.show()
        plt.close()

    return w_list, b_list, E, r, target_list1, target_list2
