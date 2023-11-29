import json
import numpy as np
import pandas as pd


# 获取指定特征x及循环寿命 [9, 10, 12, 14, 16]
def getXy_fn(file_dir, feature_list):
    data = pd.read_csv(file_dir)
    print('使用的特征为：', feature_list)
    data_x = data.iloc[:, feature_list].values
    data_y = data.iloc[:, 17].values.reshape(-1, 1)
    return data_x, data_y


# 获取指定特征x及循环寿命 ['F2', 'F3', 'F5', 'F6', 'F9']
# 或 ['D1/F1', 'V1/D2/F2', 'D3', 'D4', 'F3', 'F4', 'D5/F5', 'D6', 'F6', 'F7', 'F8', 'F9', 'Area_100_10']
def getNewXy_fn(file_path, feature_str):
    data_all = pd.read_csv(file_path)
    # print(data_all.index, data_all.columns)
    print('使用的特征为：', feature_str)
    data_x = data_all[feature_str]
    data_y = data_all[['cycle_life']]
    data_x, data_y = data_x.iloc[:, :].values, data_y.iloc[:, :].values.reshape(-1, 1)
    return data_x, data_y


# 自己提取的数据. 0是'../data/dataset.csv'  1 是 '../data/build_feature.csv'
# 0：'../data/dataset.csv'  [9, 10, 12, 13, 16]  # 2 3 5 6 9
# 1：['cell_key', 'D1/F1', 'V1/D2/F2', 'D3', 'D4', 'F3', 'F4', 'D5/F5', 'D6', 'F6', 'F7', 'F8', 'F9', 'cycle_life']
def init_data(data_path, select_feature, data_type=0):
    # 读取数据
    if data_type == 0:
        data_x, data_y = getXy_fn(data_path, select_feature)
    elif data_type == 1:
        data_x, data_y = getNewXy_fn(data_path, select_feature)
    else:
        print("data_type error")
        exit(0)

    print("y取以10为底的对数")  # 取不取对数，em效果一致。
    # ①注释掉 + convert_y='1' 使用原始数据； ②不注释+convert_y = '1'使用对数数据计算rmse；③ 不注释+convert_y = 'log10'，计算rmse还原
    data_y = np.log10(data_y)
    convert_y = '1'  # 判断是否进行还原，log10 进行还原，1不还原

    return data_x, data_y, convert_y
    pass


def getconfig(config_path):
    # 读取变量值：超参数、输出位置
    with open(config_path) as f:
        variable = json.load(f)
    # w_epsilon = variable["w_epsilon"]
    # correct = variable["correct"]

    return variable
