import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
# from scipy.stats import spearmanr
from util.data_load import getXy_fn, getNewXy_fn

# ok： 均已经测试
DATA_DIR = '../data/dataset.csv'
new_data_path = '../data/build_features.csv'
IMAGE_DIR = 'feature_images'

plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号。画图之前调用
plt.rcParams['font.family'] = ['SimSun']
np.set_printoptions(linewidth=np.inf)
np.set_printoptions(precision=6)


# 绘制特征的热力图
def plotHeatmap_fn(data_x, cycle_life, fea_names, map_type='pearson', save_dir=IMAGE_DIR, filename='heatmap.png'):
    # 计算特征和回归值之间的相关系数矩阵
    if map_type == 'pearson':
        # corr_matrix = np.corrcoef(data_x.T, cycle_life.T)  # 结果一样
        all_data = pd.DataFrame(np.column_stack((data_x, cycle_life.reshape(-1, 1))))
        corr_matrix = all_data.corr(method='pearson')
    elif map_type == 'spearman':
        all_data = pd.DataFrame(np.column_stack((data_x, cycle_life.reshape(-1, 1))))
        corr_matrix = all_data.corr(method='spearman')
    else:  # pearson
        corr_matrix = np.corrcoef(data_x.T, cycle_life.T)

    # 绘制热力图
    fig, ax = plt.subplots(figsize=(10, 8))
    x_ticks = fea_names + ['cycle_life']  # [i + 1 for i in range(data_x.shape[1] + 1)]  # np.shape(data_x)[1]
    # corr_matrix:相关系数矩阵，即要绘制热力图的数据；annot:是否在热力图上显示数值；fmt:用于指定要显示的数值的格式；
    # cmap: 指定热力图的颜色映射，coolwarm表示蓝色到红色的渐变色映射；ax: 绘图的坐标轴对象，默认为None；
    sns.heatmap(corr_matrix, annot=True, fmt=".3f", cmap="coolwarm", ax=ax, xticklabels=x_ticks, yticklabels=x_ticks)
    plt.title(map_type + " Correlation Heatmap of Features and RUL")
    plt.savefig(os.path.join(save_dir, filename))
    plt.show()


# 计算特征和rul的相关性
def plotRelationHist_fn(data_x, cycle_life, fea_names, map_type='pearson', save_dir=IMAGE_DIR, filename='features.png'):
    # 计算特征和回归值之间的相关系数矩阵
    if map_type == 'pearson':
        # corr_matrix = np.corrcoef(data_x.T, cycle_life.T)  # 结果一样
        all_data = pd.DataFrame(np.column_stack((data_x, cycle_life.reshape(-1, 1))))
        corr_matrix = all_data.corr(method='pearson')
    elif map_type == 'spearman':
        all_data = pd.DataFrame(np.column_stack((data_x, cycle_life.reshape(-1, 1))))
        corr_matrix = all_data.corr(method='spearman')
    else:  # pearson
        corr_matrix = np.corrcoef(data_x.T, cycle_life.T)

    # 1. 相关系数：最后一行的 前n-1个
    cor_vector = corr_matrix.iloc[-1, :-1].values
    print(map_type + ' Correlation: ', cor_vector)

    # 方法2：计算斯皮尔曼系数
    # cols = np.shape(data_x)[1]  # 特征的数量
    # cor_vector = []  # 存储每个特征与 RUL 之间的斯皮尔曼相关系数。
    # for col in range(cols):
    #     # 返回一个包含相关系数和 p 值的元组，但只取相关系数部分，即 [0] 索引。
    #     cor_vector.append(spearmanr(data_x[:, col], cycle_life)[0])
    # cor_vector = np.asarray(cor_vector)
    # print("Spearman Correlation:", cor_vector)

    # 2. 绘制图像
    show_cors = [round(x, 3) for x in cor_vector]  # 保留三位小数
    plt.figure(figsize=(12, 10))  # 要求柱体宽度参数是一个标量值
    plt.bar(fea_names, show_cors)
    plt.xticks(fontsize=12)
    plt.ylim([-1, 1])  # 设置y轴开始结束
    for i in range(len(fea_names)):
        if show_cors[i] < 0:
            plt.text(fea_names[i], show_cors[i], str(show_cors[i]), ha='center', va='top', fontsize=12)
        else:
            plt.text(fea_names[i], show_cors[i], str(show_cors[i]), ha='center', va='bottom', fontsize=12)
    # 倾斜显示横轴刻度标签
    plt.xticks(rotation=45, ha='right')
    # 保存的图片是空白的，绘图操作没有完成
    plt.savefig(os.path.join(save_dir, filename))
    plt.show()

    # 3. 进行排序
    sort_index = np.argsort(-np.abs(cor_vector))  # np.abs()是从小到大， 加负号是从大到小
    new_fea_names = [fea_names[i] for i in sort_index]
    new_rul_feature_cors = cor_vector[sort_index]
    print('sorted features:  ', new_fea_names)
    print('sorted relations: ', new_rul_feature_cors)

    return new_fea_names, new_rul_feature_cors


# 未验证
def greyRelation_fn(data_x, cycle_life, fea_names, save_dir=IMAGE_DIR, filename='grey_correlation.png'):
    all_data = np.column_stack((data_x, cycle_life.reshape(-1, 1)))
    normalized_data = (all_data - all_data.min(axis=0)) / (all_data.max(axis=0) - all_data.min(axis=0))

    n_features = all_data.shape[1]
    relation_matrix = np.zeros((n_features, n_features))

    # 计算特征间和特征与cycle_life之间的关联度
    for i in range(n_features):
        for j in range(i, n_features):  # 特征与cycle_life的关联度
            temp = np.abs(normalized_data[:, i] - normalized_data[:, j])  # n*1
            rho = 0.5  # 分辨系数
            relation = np.mean((1 - temp) / (1 + rho * temp))
            # mink = np.min(temp)
            # maxk = np.max(temp)
            # if i == j:
            #     relation = 1
            # else:
            #     relation = np.mean((mink + rho * maxk) / (temp + rho * maxk))
            relation_matrix[i, j] = relation_matrix[j, i] = relation

    # 可视化关联度矩阵
    x_ticks = fea_names + ['cycle_life']
    plt.figure(figsize=(12, 10))
    sns.heatmap(relation_matrix, annot=True, cmap='coolwarm', xticklabels=x_ticks, yticklabels=x_ticks)
    plt.title('Grey Correlation Matrix')

    # 保存图表
    plt.savefig(f'{save_dir}/{filename}')
    plt.show()
    plt.close()
    return relation_matrix


if __name__ == '__main__':
    # 1 表示使用指定特征和原始的y的相关系数和特征图。 2表示选取指定特征 与 log10(y)的关系
    test = 4
    if test == 1:
        select_feature = range(8, 17)
        print(select_feature)
        feature_names = [str(i) for i in select_feature]
        dataX, dataY = getXy_fn(DATA_DIR, select_feature)

        plotRelationHist_fn(dataX, dataY, feature_names, filename='all_9features_y.png')
        plotHeatmap_fn(dataX, dataY, feature_names, filename='all_9features_y_heatmap.png')
    elif test == 2:
        select_feature = [f'F{i}' for i in range(1, 10)]  # ['F2', 'F3', 'F5', 'F6', 'F9']
        print(select_feature)
        feature_names = select_feature
        dataX, dataY = getNewXy_fn(DATA_DIR, select_feature)
        dataY = np.log10(dataY)

        plotRelationHist_fn(dataX, dataY, feature_names, filename='all_F_features_log10_y.png')
        plotHeatmap_fn(dataX, dataY, feature_names, filename='all_F_features_log10_y_heatmap.png')

    elif test == 3:
        # ['cell_key', 'D1/F1', 'V1/D2/F2', 'D3', 'D4', 'F3', 'F4', 'D5/F5', 'D6', 'F6', 'F7', 'F8', 'F9', 'cycle_life']
        select_feature = ['V1/D2/F2', 'F3', 'D5/F5', 'F6', 'F9']  # 'V1/D2/F2'
        print(select_feature)
        dataX, dataY = getNewXy_fn(new_data_path, select_feature)
        dataY = np.log10(dataY)

        plotRelationHist_fn(dataX, dataY, select_feature, filename='F_five_features_log10_y.png')
        plotHeatmap_fn(dataX, dataY, select_feature, filename='F_five_features_log10_y_heatmap.png')

    elif test == 4:
        select_feature = ['D1/F1', 'V1/D2/F2', 'D3', 'D4', 'F3', 'F4', 'D5/F5', 'D6', 'F6', 'F7', 'F8', 'F9',
                          'Area_100_10']
        dataX, dataY = getNewXy_fn(new_data_path, select_feature)
        dataY = np.log10(dataY)

        # plotHeatmap_fn(dataX, dataY, select_feature, map_type='pearson', filename='13features_logy_pearson.png')
        # plotHeatmap_fn(dataX, dataY, select_feature, map_type='spearman', filename='13features_logy_spearman.png')
        plotRelationHist_fn(dataX, dataY, select_feature, map_type='pearson', filename='13features_logy_pearson_v.png')
        plotRelationHist_fn(dataX, dataY, select_feature, map_type='spearman', filename='13features_logy_spearman_v.png')
        # greyRelation_fn(dataX, dataY, select_feature, filename='13features_logy_greyMap1.png')

    pass

'''
pearson:  'V1/D2/F2', 'D1/F1', 'Area_100_10', 'F6', 'F3', 'F8', 'D3', 'F9', 'F7', 'F4', 'D4', 'D5/F5', 'D6'
spearman: 'V1/D2/F2', 'Area_100_10', 'D1/F1', 'F8', 'D3', 'D6', 'F9', 'F7', 'F4', 'D4', 'F3', 'D5/F5', 'F6'
→ F2, Area, F6 F8 D3
pearson Correlation:  [-0.881721 -0.912417  0.21731   0.085992  0.404514 -0.108369 -0.046489  0.036996  0.502563  0.17608  -0.332459  0.204995 -0.819641]
sorted features:   ['V1/D2/F2', 'D1/F1', 'Area_100_10', 'F6', 'F3', 'F8', 'D3', 'F9', 'F7', 'F4', 'D4', 'D5/F5', 'D6']
sorted relations:  [-0.912417 -0.881721 -0.819641  0.502563  0.404514 -0.332459  0.21731   0.204995  0.17608  -0.108369  0.085992 -0.046489  0.036996]
spearman Correlation:  [-0.848095 -0.873565  0.300565 -0.154441  0.091488 -0.158596 -0.081052 -0.258449  0.007478  0.201168 -0.643258  0.218499 -0.856168]
sorted features:   ['V1/D2/F2', 'Area_100_10', 'D1/F1', 'F8', 'D3', 'D6', 'F9', 'F7', 'F4', 'D4', 'F3', 'D5/F5', 'F6']
sorted relations:  [-0.873565 -0.856168 -0.848095 -0.643258  0.300565 -0.258449  0.218499  0.201168 -0.158596 -0.154441  0.091488 -0.081052  0.007478]'''
