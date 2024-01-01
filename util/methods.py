import numpy as np
from sklearn.preprocessing import StandardScaler


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
def ls_fn(x_new, y_new):
    std_w = np.linalg.inv(x_new.T.dot(x_new)).dot(x_new.T).dot(y_new)
    return std_w


# 求模型参数w,  x y 在 tls内部未进行标准化  w未进行还原
def tls_fn1(std_x, std_y):
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


def tls_fn(std_X, std_Y):
    # 标准化X_train
    # standard_X = np.std(x_train, axis=0).reshape(-1, 1)
    # stand_scaler = StandardScaler()
    # std_X = stand_scaler.fit_transform(x_train)
    # mean_X = stand_scaler.mean_.reshape(-1, 1)

    # # 标准化Y_train
    # mean_Y = np.array(np.mean(y_train)).reshape(-1, 1)
    # standard_Y = np.std(y_train, axis=0).reshape(-1, 1)
    # std_Y = (y_train - mean_Y) / standard_Y

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
    # W = np.dot(std_W, standard_Y) / standard_X
    # # 计算b
    # _ = 0
    # for i in range(n):
    #     _ = _ + std_W[i] * mean_X[i] * standard_Y / standard_X[i]
    # b = mean_Y - _

    return std_W
