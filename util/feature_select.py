import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


# ok： 均已经测试
DATA_DIR = '../data/dataset.csv'
IMAGE_DIR = 'feature_images'


plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号。画图之前调用
plt.rcParams['font.family'] = ['SimSun']


# 获取指定特征x及循环寿命 [9, 10, 12, 14, 16]
def getXy_fn(file_dir, feature_list):
    data = pd.read_csv(file_dir)
    print('使用的特征为：', feature_list)
    data_x = data.iloc[:, feature_list].values
    data_y = data.iloc[:, 17].values.reshape(-1, 1)
    return data_x, data_y


# 获取指定特征x及循环寿命 ['F2', 'F3', 'F5', 'F6', 'F9']
def getXyByStr_fn(file_dir, feature_str):
    data_all = pd.read_csv(file_dir)
    print('使用的特征为：', feature_str)
    data_x = data_all[feature_str]
    data_y = data_all[['cyclelife']]
    data_x, data_y = data_x.iloc[:, :].values, data_y.iloc[:, :].values.reshape(-1, 1)
    return data_x, data_y


# 计算特征和rul的相关性
def calFeaRulCorrelation_fn(data_x, cycle_life, fea_name, save_dir=IMAGE_DIR, filename='features.png'):
    # 1. 计算斯皮尔曼系数
    cols = np.shape(data_x)[1]  # 特征的数量
    rul_feature_cors = []  # 存储每个特征与 RUL 之间的斯皮尔曼相关系数。
    for col in range(cols):
        # 返回一个包含相关系数和 p 值的元组，但只取相关系数部分，即 [0] 索引。
        rul_feature_cors.append(spearmanr(data_x[:, col], cycle_life)[0])
    rul_feature_cors = np.asarray(rul_feature_cors)
    print("Spearman Correlation:", rul_feature_cors)

    # 2. 绘制图像
    show_cors = [round(x, 3) for x in rul_feature_cors]  # 保留三位小数
    plt.figure(figsize=(12, 10))  # 要求柱体宽度参数是一个标量值
    plt.bar(fea_name, show_cors)
    plt.xticks(fontsize=12)
    plt.ylim([-1, 1])  # 设置y轴开始结束
    for i in range(len(fea_name)):
        if show_cors[i] < 0:
            plt.text(fea_name[i], show_cors[i], str(show_cors[i]), ha='center', va='top', fontsize=12)
        else:
            plt.text(fea_name[i], show_cors[i], str(show_cors[i]), ha='center', va='bottom', fontsize=12)
    # 倾斜显示横轴刻度标签
    plt.xticks(rotation=45, ha='right')
    # 保存的图片是空白的，绘图操作没有完成
    plt.savefig(os.path.join(save_dir, filename))
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
def plotHeatMap_fn(data_x, cycle_life, save_dir=IMAGE_DIR, filename='correlation_heatmap.png'):
    plt.rcParams['axes.unicode_minus'] = False
    # 获取特征和回归值之间的相关系数矩阵
    corr_matrix = np.corrcoef(data_x.T, cycle_life.T)
    # 绘制热力图
    fig, ax = plt.subplots(figsize=(10, 8))
    x_ticks = [i + 1 for i in range(data_x.shape[1] + 1)]  # np.shape(data_x)[1]
    # corr_matrix:相关系数矩阵，即要绘制热力图的数据；annot:是否在热力图上显示数值；fmt:用于指定要显示的数值的格式；
    # cmap: 指定热力图的颜色映射，coolwarm表示蓝色到红色的渐变色映射；ax: 绘图的坐标轴对象，默认为None；
    sns.heatmap(corr_matrix, annot=True, fmt=".3f", cmap="coolwarm", ax=ax, xticklabels=x_ticks, yticklabels=x_ticks)
    plt.title("Correlation Heatmap of Features and RUL")
    plt.xlabel("Column")
    plt.ylabel("Column")
    plt.savefig(os.path.join(save_dir, filename))
    plt.show()


if __name__ == '__main__':
    # 1 表示使用指定特征和原始的y的相关系数和特征图。 2表示选取指定特征 与 log10(y)的关系
    test = 2
    if test == 1:
        select_feature = range(8, 17)
        print(select_feature)
        feature_names = [str(i) for i in select_feature]
        dataX, dataY = getXy_fn(DATA_DIR, select_feature)

        calFeaRulCorrelation_fn(dataX, dataY, feature_names, filename='all_9features_y.png')
        plotHeatMap_fn(dataX, dataY, filename='all_9features_y_heatmap.png')
    else:
        select_feature = [f'F{i}' for i in range(1, 10)]   # ['F2', 'F3', 'F5', 'F6', 'F9']
        print(select_feature)
        feature_names = select_feature
        dataX, dataY = getXyByStr_fn(DATA_DIR, select_feature)
        dataY = np.log10(dataY)

        calFeaRulCorrelation_fn(dataX, dataY, feature_names, filename='all_F_features_log10_y.png')
        plotHeatMap_fn(dataX, dataY, filename='all_F_features_log10_y_heatmap.png')

    pass
