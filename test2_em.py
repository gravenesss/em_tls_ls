from sklearn.model_selection import train_test_split
from now_utils import *
from util.data_load import init_data, getconfig


# 测试 em 算法
def test_em(test_ratio, cur_seed, plot_flag=True):
    train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=test_ratio, random_state=cur_seed)
    # em 结果 em中就可以确定是否还原y 计算rmse
    em_err, tls_err = emTest_fn(train_x, train_y, test_x, test_y, cur_seed, w_epsilon, correct, plot_pic=plot_flag)

    global count_em
    if em_err < tls_err:
        count += 1
        print("yes")

    pass


if __name__ == '__main__':
    data_path = 'data/build_features.csv'
    # 'V1/D2/F2', 'D1/F1', 'Area_100_10', 'F6', 'F3', 'F8', 'D3', 'F9', 'F7', 'F4', 'D4', 'D5/F5', 'D6'
    # select_feature = ['V1/D2/F2', 'Area_100_10', 'F6', 'F8', 'D3']  # → F2, Area, F6 F8 D3
    select_feature = ['V1/D2/F2', 'F3', 'F6', 'F9', 'Area_100_10']  # 'D5/F5', 'Area_100_10'
    data_x, data_y, convert_y = init_data(data_path, select_feature, 1)  # 全局使用
    variable = getconfig('config.json')
    w_epsilon = variable['w_epsilon']
    correct = variable['correct']

    # 噪声模式固定，查看随机划分数据集 和 随机噪声的影响。
    count_em = 0
    for i in range(0, 10):
        print(f"seed {i:02d}=================================")
        test_em(0.1, i, True)
    print(count_em)

    pass
