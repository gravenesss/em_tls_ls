import os
from datetime import datetime

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv

from sklearn.linear_model import LinearRegression, ElasticNet, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr

"""0. 读取数据，查看特征相关性ok ------------------------------------------------------------------------"""


# IMAGE_DIR = '../images'
# CSV_DIR = '../csv'

# 获取指定特征x及循环寿命 [9, 10, 12, 14, 16]
def get_feature(feature_list):
    data = pd.read_csv('../data/dataset.csv')
    print('使用的特征为：', feature_list)
    data_x = data.iloc[:, feature_list].values
    data_y = data.iloc[:, 17].values.reshape(-1, 1)
    return data_x, data_y


# 获取指定特征x及循环寿命 ['F2', 'F3', 'F5', 'F6', 'F9']
# 如果要和get_feature返回的结果一致： data_x, data_y = data_x.iloc[,],
def get_xy_by_str(feature_str):
    data_all = pd.read_csv('../data/dataset.csv')
    print('使用的特征为：', feature_str)
    data_x = data_all[feature_str]
    data_y = data_all[['cyclelife']]
    data_x, data_y = data_x.iloc[:, :].values, data_y.iloc[:, :].values.reshape(-1, 1)
    return data_x, data_y


# 计算特征和rul的相关性
def cal_fea_rul_correlation(data_x, RUL, fea_name, save_dir='../images'):
    # 1. 计算斯皮尔曼系数
    cols = np.shape(data_x)[1]  # 特征的数量
    rul_feature_cors = []  # 存储每个特征与 RUL 之间的斯皮尔曼相关系数。
    for col in range(cols):
        # 返回一个包含相关系数和 p 值的元组，但只取相关系数部分，即 [0] 索引。
        rul_feature_cors.append(spearmanr(data_x[:, col], RUL)[0])
    rul_feature_cors = np.asarray(rul_feature_cors)
    print("Spearman Correlation:", rul_feature_cors)

    # 2. 绘制图像
    show_cors = [round(x, 3) for x in rul_feature_cors]  # 保留三位小数
    plt.figure(figsize=(12, 10))  # 要求柱体宽度参数是一个标量值
    plt.bar(fea_name, show_cors)
    plt.xticks(fontsize=8)
    plt.ylim([-1, 1])  # 设置y轴开始结束
    for i in range(len(fea_name)):
        if show_cors[i] < 0:
            plt.text(fea_name[i], show_cors[i], str(show_cors[i]), ha='center', va='top', fontsize=8)
        else:
            plt.text(fea_name[i], show_cors[i], str(show_cors[i]), ha='center', va='bottom', fontsize=8)
    # 倾斜显示横轴刻度标签
    plt.xticks(rotation=45, ha='right')
    # 保存的图片是空白的，绘图操作没有完成
    plt.savefig(os.path.join(save_dir, 'all_16_features.png'))
    plt.show()

    # 3. 进行排序
    # 创建 (x, y) 对的列表
    xy_pairs = list(zip(fea_name, rul_feature_cors))
    # 按照 y 的绝对值大小 从大到小排序
    sorted_list = sorted(xy_pairs, key=lambda item: abs(item[1]), reverse=True)
    # 提取排序后的 y
    sorted_y = [item[1] for item in sorted_list]
    # 提取排序后的 (x, y) 对
    sorted_xy_pairs = [(item[0], item[1]) for item in sorted_list]
    print(sorted_y)
    print(sorted_xy_pairs)

    return rul_feature_cors, sorted_xy_pairs


# 绘制特征的热力图
def plot_head_map(data_x, RUL):
    plt.rcParams['axes.unicode_minus'] = False
    # 获取特征和回归值之间的相关系数矩阵
    corr_matrix = np.corrcoef(data_x.T, RUL.T)
    # 绘制热力图
    fig, ax = plt.subplots(figsize=(10, 8))
    x_ticks = [i + 1 for i in range(data_x.shape[1] + 1)]  # np.shape(data_x)[1]
    # corr_matrix:相关系数矩阵，即要绘制热力图的数据；annot:是否在热力图上显示数值；fmt:用于指定要显示的数值的格式；
    # cmap: 指定热力图的颜色映射，coolwarm表示蓝色到红色的渐变色映射；ax: 绘图的坐标轴对象，默认为None；
    sns.heatmap(corr_matrix, annot=True, fmt=".3f", cmap="coolwarm", ax=ax, xticklabels=x_ticks, yticklabels=x_ticks)
    plt.title("Correlation Heatmap of Features and RUL")
    plt.xlabel("Features")
    plt.ylabel("RUL")
    plt.show()


"""1. tls: 进行还原 / 不进行还原(之后通过w_std获取w b)----------------------------------------------------"""


def ls(x_new, y_new):  # X_train, Y_train
    # 求模型参数,Y=WX+b (X^T X)^- X^T y
    std_W = np.dot(np.dot(np.linalg.inv(np.dot(x_new.T, x_new)), x_new.T), y_new)

    return std_W


# 师哥的tls  x y 在 tls内部未进行标准化  w未进行还原
def tls2(std_X, std_Y):
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
def get_wb(w_std, eta, m, x_mean, y_mean):
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


# 根据x w b y获取均方根误差
def get_rmse_Loss_by_wb(x_test, y_test, w, b=0.0):
    y_predict = np.dot(x_test, w) + b
    mse = mean_squared_error(y_test, y_predict)
    rmse = mse ** 0.5
    # 计算 perr
    n = y_test.size
    perr = 0
    for i in range(n):
        perr += np.abs(y_test[i] - y_predict[i]) / y_test[i]
    perr = (perr * 100 / n)[0]
    return rmse, perr


def get_rmse_loss_restore(x_test, y_test, w, b=0.0, convert_y='1'):
    y_predict = np.dot(x_test, w) + b
    #  ** 运算符或内置函数 pow() 来实现幂运算
    if convert_y == 'log10':  # log10(y) = y' y=10^y'
        # print("y_predict1:", y_predict, '\ny_test1:', y_test, '\n')
        y_predict = np.power(10, y_predict)
        y_test = np.power(10, y_test)
        # print("y_predict2:", y_predict, '\ny_test2:', y_test, '\n\n')
    elif convert_y == 'loge':
        y_predict = np.exp(y_predict)
        y_test = np.exp(y_test)
    else:
        pass
    rmse = np.sqrt(mean_squared_error(y_test, y_predict))
    # 计算 perr
    n = y_test.size
    perr = 0
    for i in range(n):
        perr += np.abs(y_test[i] - y_predict[i]) / y_test[i]
    perr = (perr * 100 / n)[0]
    return rmse, perr


# 计算标准差，均值为0的情况，返回matrix每列的标准差 ok
def calculate_std(matrix):
    n, m = matrix.shape[0], 1 if matrix.ndim == 1 else matrix.shape[1]
    mean_col = 0  # 设置为0而不是原来的均值 np.mean(matrix, axis=0)
    diff_matrix = (matrix - mean_col) ** 2
    variance = np.sum(diff_matrix, axis=0) / n
    std = np.sqrt(variance)
    return std


# 绘制图像，x ，y1， y2
def plot_xyy(x, y1, y2, title, y_label, x_label, file_dir, file_name):
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.family'] = ['SimSun']
    plt.plot(x, y1, label='tls', marker='o')
    plt.plot(x, y2, label='em&tls', marker='^')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.savefig(os.path.join(file_dir, file_name))
    plt.show()


def plot_xys(x, ys, labels, markers, x_label, y_label, file_dir, file_name, title='RMSE'):
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号。画图之前调用
    plt.rcParams['font.family'] = ['SimSun']

    for i in range(len(ys)):
        plt.plot(x, ys[i], label=labels[i], marker=markers[i])
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.savefig(os.path.join(file_dir, file_name))
    plt.show()


# 绘制图像，x ，y1， y2
def plot_xyy11(x, y1, y2, title, y_label, x_label, file_name):
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号。画图之前调用
    plt.rcParams['font.family'] = ['SimSun']
    plt.plot(x, y1, label='tls', marker='o')
    plt.plot(x, y2, label='em&tls', marker='^')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.savefig(os.path.join('../images', file_name))
    plt.show()


def plot_xwb(sequence, mid_tls_wb, mid_em_wb, x_label, feature_len, file_dir, file_name):
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号。画图之前调用
    fig, axes = plt.subplots(3, 2, figsize=(10, 10))
    for i, ax in enumerate(axes.flatten()):  # 遍历每个子图 0~6, 最后一个是b
        y1 = mid_tls_wb[:, i]
        y2 = mid_em_wb[:, i]
        title = 'b' if i == feature_len else 'w' + str(i + 1)
        ax.plot(sequence, y1, label='tls', marker='o')
        ax.plot(sequence, y2, label='em', marker='^')
        ax.set_xlabel(x_label)
        ax.set_ylabel('The value of ' + title)
        # ax.set_title(title)   # 设置子图标题
        ax.legend()  # 添加图例
    plt.tight_layout()  # 调整子图布局的间距
    plt.savefig(os.path.join(file_dir, file_name))
    plt.show()


def plot_x_wbs(sequence, ys, labels, markers, x_label, feature_len, file_dir, file_name):
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号。画图之前调用
    fig, axes = plt.subplots(3, 2, figsize=(10, 10))
    for i, ax in enumerate(axes.flatten()):  # 遍历每个子图 0~6, 最后一个是b
        y_label = 'b' if i == feature_len else 'w' + str(i + 1)

        for j, y in enumerate(ys):
            ax.plot(sequence, y[:, i], label=labels[j], marker=markers[j])

        ax.set_xlabel(x_label)
        ax.set_ylabel('The value of ' + y_label)
        # ax.set_title(title)   # 设置子图标题
        ax.legend()  # 添加图例
    plt.tight_layout()  # 调整子图布局的间距
    plt.savefig(os.path.join(file_dir, file_name))
    plt.show()


# em 实验1
def em_tls(x_now, y_now, m, x_std, y_std, x_mean, y_mean, x_test, y_test, w_dis_epsilon=1e-6, correct=1e-2):
    # 假设样本误差E的每一列符合N(0,1) 和 标签误差r符合N(0,1)
    diag_x, diag_x_inv = np.eye(m), np.eye(m)
    flag = True
    w_pre = None
    # 记录 w 和 rmse
    wb_list = []
    rmse_list = []

    while flag:
        # 1.计算w
        w1 = tls2(np.dot(x_now, diag_x), y_now)  # x'=x*sigma_x  w'=sigma_x^(-1)*w  w=sigma_x*w'
        w_std = np.dot(diag_x, w1)  # m*m * m*1
        # 还原 w 计算 rmse
        w_original, b_original = get_wb(w_std, y_std / x_std, m, x_mean, y_mean)
        rmse, _ = get_rmse_Loss_by_wb(x_test, y_test, w_original, b_original)
        wb_list.append(np.vstack((w_original, b_original)))
        rmse_list.append(rmse)

        # 2.根据 w、diag_x 计算 E 和 r
        w_t = np.transpose(w_std).reshape(1, -1)
        tmp_x = diag_x_inv
        tmp_x = np.dot(tmp_x, tmp_x)
        denominator = np.dot(np.dot(w_t, tmp_x), w_std) + 1  # wt: 1*m tmp_x:m*m  w:m*1
        r_up = (np.dot(x_now, w_std) - y_now).reshape(-1, 1)  # n*m * m*1 => n*1
        r = r_up / denominator  # 1*1
        E_up = -np.dot(np.dot(r_up, w_t), tmp_x)  # n*1 * 1*m * m*m => n*m
        E = E_up / denominator

        # 3.更新sigma_x：根据样本误差的方差 和 标签误差的方差
        E_std = calculate_std(E)
        r_std = calculate_std(r)
        if np.any(E_std == 0):
            print("样本误差 的标准差某一列存在为0的情况")
            break
        if np.any(r_std == 0):
            print("标签误差 的标准差某一列存在为0的情况")
            break
        eta = (r_std + correct) / (E_std + correct)
        eta_inv = (E_std + correct) / (r_std + correct)
        for i in range(m):
            diag_x[i][i] = eta[i]
            diag_x_inv[i][i] = eta_inv[i]

        # 如果两次迭代的参数差距小于 w_dis_epsilon 则结束循环
        if w_pre is None:
            w_pre = w_std
        else:
            gap = np.linalg.norm(w_std - w_pre)  # 欧氏距离
            w_pre = w_std
            flag = False if gap <= w_dis_epsilon else True

    # print(rmse_list, '\n', wb_list, '\n')
    # plt.plot([x+1 for x in range(len(rmse_list))], rmse_list)
    # plt.show()
    # 要根据 rmse_list 排序，需要记录
    sorted_data = sorted(zip(rmse_list, wb_list))
    mid_rmse, mid_wb = sorted_data[len(sorted_data) // 2]
    # midia_rmse = np.median(new_rmses)
    return mid_rmse, mid_wb


# em 实验2
def em_tls1(x_now, y_now, m, x_std, y_std, x_mean, y_mean, x_test, y_test, w_dis_epsilon=1e-6, correct=1e-2, convert_y='1'):
    # 假设样本误差E的每一列符合N(0,1) 和 标签误差r符合N(0,1)
    diag_x, diag_x_inv = np.eye(m), np.eye(m)
    flag = True
    w_pre = None
    # 记录 w 和 rmse
    wb_list = []
    rmse_list = []

    while flag:
        # 1.计算w
        w1 = tls2(np.dot(x_now, diag_x), y_now)  # 师哥的tls  x'=x*sigma_x  w'=sigma_x^(-1)*w  w=sigma_x*w'
        w_std = np.dot(diag_x, w1)  # m*m * m*1
        w_t = np.transpose(w_std).reshape(1, -1)
        # 还原 w 计算 rmse
        w_original, b_original = get_wb(w_std, y_std / x_std, m, x_mean, y_mean)
        rmse, _ = get_rmse_Loss_by_wb(x_test, y_test, w_original, b_original)
        wb_list.append(np.vstack((w_original, b_original)))
        rmse_list.append(rmse)

        # 2.根据 w、diag_x 计算 E 和 r
        tmp_x = diag_x_inv
        tmp_x = np.dot(tmp_x, tmp_x)
        denominator = np.dot(np.dot(w_t, tmp_x), w_std) + 1  # wt: 1*m tmp_x:m*m  w:m*1
        r_up = (np.dot(x_now, w_std) - y_now).reshape(-1, 1)  # n*m * m*1 => n*1
        r = r_up / denominator  # 1*1
        E_up = -np.dot(np.dot(r_up, w_t), tmp_x)  # n*1 * 1*m * m*m => n*m
        E = E_up / denominator

        # 3.更新sigma_x：根据样本误差的方差 和 标签误差的方差
        E_std = calculate_std(E)
        r_std = calculate_std(r)
        if np.any(E_std == 0):
            print("样本误差 的标准差某一列存在为0的情况")
            break
        if np.any(r_std == 0):
            print("标签误差 的标准差某一列存在为0的情况")
            break
        eta = (r_std + correct) / (E_std + correct)
        eta_inv = (E_std + correct) / (r_std + correct)
        for i in range(m):
            diag_x[i][i] = eta[i]
            diag_x_inv[i][i] = eta_inv[i]

        # 如果两次迭代的参数差距小于 w_dis_epsilon 则结束循环
        if w_pre is None:
            w_pre = w_std
        else:
            gap = np.linalg.norm(w_std - w_pre)  # 欧氏距离
            w_pre = w_std
            flag = False if gap <= w_dis_epsilon else True

    # print(rmse_list, '\n', wb_list, '\n')
    # plt.plot([x+1 for x in range(len(rmse_list))], rmse_list)
    # plt.show()
    # 要根据 rmse_list 排序，需要记录
    sorted_data = sorted(zip(rmse_list, wb_list))
    mid_rmse, mid_wb = sorted_data[len(sorted_data) // 2]

    mid_rmse, _ = get_rmse_loss_restore(x_test, y_test, mid_wb[0:5], mid_wb[5], convert_y)
    # midia_rmse = np.median(new_rmses)
    return mid_rmse, mid_wb


""" 2. 保存运行结果 / 读取运行结果并绘制图像 ------------------------------------------------------------------------ """


# 保存 x, y.... x_labels, y_labels 到
def save_csvs(x, ys, x_labels, y_labels, comments, file_dir, file_name):
    data_all = [x] + ys
    header = [x_labels] + y_labels
    # df = pd.DataFrame()
    # df[x_labels] = x
    # for i, label in enumerate(y_labels):
    #     df[label] = ys[i]
    df = pd.DataFrame(data_all, header)

    # 获取当前时间作为文件名==== 以下相同
    filename = os.path.join(file_dir, file_name)
    # 保存到 csv 文件     # df.to_csv(filename, index=False)  # 如果没有备注信息时
    with open(filename, 'w', newline='', encoding='utf-8-sig') as file:
        # 使用 utf8编码，并添加了-sig以确保在Excel中正确显示中文字符
        writer = csv.writer(file)
        # 写入注解
        writer.writerow(['##COMMENT_START##'])
        for comment in comments:
            writer.writerow([comment])
        writer.writerow(['##COMMENT_END##'])
        writer.writerow(header)
        writer.writerows(df.values)


# 编写备注信息 和 训练生成的数据到csv文件
def save_csv(x, y1, y2, y3, y4, comments, file_dir, file_name):
    header = ['x', 'tls_rmse', 'em_rmse', 'tls_wb', 'em_wb']
    # 创建 DataFrame
    df = pd.DataFrame({'x': x, 'tls_rmse': y1, 'em_rmse': y2, 'tls_wb': y3, 'em_wb': y4})

    # 获取当前时间作为文件名
    filename = os.path.join(file_dir, file_name)
    # 保存到 csv 文件     # df.to_csv(filename, index=False)  # 如果没有备注信息时
    with open(filename, 'w', newline='', encoding='utf-8-sig') as file:
        # 使用 utf8编码，并添加了-sig以确保在Excel中正确显示中文字符
        writer = csv.writer(file)
        # 写入注解
        writer.writerow(['##COMMENT_START##'])
        for comment in comments:
            writer.writerow([comment])
        writer.writerow(['##COMMENT_END##'])
        writer.writerow(header)
        writer.writerows(df.values)


# 编写备注信息 和 训练生成的数据到csv文件  pre
def save_csv11(x, y1, y2, y3, y4, comments, prefix):
    """ 给定的数据
    comments = ['# 训练的参数：(21:39 => 12:15:19)', '训练集比例：0.2 => 0.9(步长0.05)',
                '随机生成噪声比例次数：100', '随机划分数据集次数：100',
                '随机生成噪声次数：50']
    x = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    y1 = [0.3876467639270337, 0.3490801368178834, 0.2929356873686641, 0.2561889404502434, 0.25366105161447383,
          0.21867789591936665, 0.21878395407547196, 0.207509891097104, 0.18248199795320802, 0.17476602819870285,
          0.17350623473424187, 0.17073918767523058, 0.17144587543209344, 0.16180962212532388, 0.14725045391530728]
    y2 = [0.19867801324210893, 0.17945301032886535, 0.15773748683812966, 0.14975912295745547, 0.1464498630344859,
          0.13951489954822716, 0.1390783838671, 0.1363065287744576, 0.13033812007622733, 0.12836308766441845,
          0.12941284905072165, 0.1282258634764192, 0.12825218670420238, 0.1261821833333604, 0.1200409820328712]
    """
    header = ['x', 'tls_rmse', 'em_rmse', 'tls_wb', 'em_wb']
    # 创建 DataFrame
    df = pd.DataFrame({'x': x, 'tls_rmse': y1, 'em_rmse': y2, 'tls_wb': y3, 'em_wb': y4})

    # 获取当前时间作为文件名
    filename = os.path.join('../csv', prefix + "-" + datetime.now().strftime("%Y%m%d%H%M%S") + '.csv')
    # 保存到 csv 文件     # df.to_csv(filename, index=False)  # 如果没有备注信息时
    with open(filename, 'w', newline='', encoding='utf-8-sig') as file:
        # 使用 utf8编码，并添加了-sig以确保在Excel中正确显示中文字符
        writer = csv.writer(file)
        # 写入注解
        writer.writerow(['##COMMENT_START##'])
        for comment in comments:
            writer.writerow([comment])
        writer.writerow(['##COMMENT_END##'])
        writer.writerow(header)
        writer.writerows(df.values)


# 读取保存的csv文件中的 rmse 进行绘制， 可以附加绘制 wb 的情况
def plot_csv(filename):
    # 报错：libiomp5md.dll
    # https://blog.csdn.net/peacefairy/article/details/110528012
    # 读取 CSV 文件
    # df = pd.read_csv(filename)
    with open(filename, 'r', encoding='utf-8-sig') as file:
        data = file.readlines()
    # 解析备注信息和数据内容
    comments_start = data.index('##COMMENT_START##\n')
    comments_end = data.index('##COMMENT_END##\n')
    comments = [line.strip() for line in data[comments_start + 1:comments_end]]

    # 1、获取列的信息
    header = data[comments_end + 1].strip().split(',')
    # 获取具体的数据
    content = [line.strip().split(',') for line in data[comments_end + 2:]]
    # 创建DataFrame
    df = pd.DataFrame(content, columns=header)
    x = df['x'].astype(float)
    y1 = df['tls_rmse'].astype(float)
    y2 = df['em_rmse'].astype(float)

    # 2、绘制折线图 在折点处添加圆圈
    plt.rcParams['font.family'] = ['SimSun']
    # 折的标记：https://matplotlib.org/stable/api/markers_api.html
    plt.plot(x, y1, label='tls', marker='o')
    plt.plot(x, y2, label='em&tls', marker='^')
    # 添加标题和标签
    plt.xlabel('Proportion of Training Data')
    plt.ylabel('Test RMSE')
    # 添加图例
    plt.legend()
    plt.show()


""" 3. 其他模型进行训练 ------------------------------------------------------------------------ """


# 模型训练、也进行了还原
def model_rmse(x, y, x_test, y_test, model_name, convert_y='1', poly=2):
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
