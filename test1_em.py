import copy
import os
import pandas as pd
from now_utils import *

np.set_printoptions(linewidth=np.inf)  # 设置ndaary一行显示，不折行
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号。画图之前调用
plt.rcParams['font.family'] = ['SimSun']


# 测试 em 算法
def test_em(cur_seed, plot_flag=True, save_flag=True, plot_stds_wb=True):
    train_num = round(data.shape[0] * train_ratio)
    copy_data = copy.deepcopy(data)

    # 1、随机排序
    np.random.seed(cur_seed)
    new_data = np.random.permutation(copy_data)
    x_train_random = new_data[:train_num, :-1]
    y_train_random = np.log10(new_data[:train_num, -1]).reshape(-1, 1)
    x_test_random = new_data[train_num:, :-1]
    y_test_random = np.log10(new_data[train_num:, -1]).reshape(-1, 1)

    # 2、进行训练：LS
    w_ls, b_ls = lsOrTls_fn(x_train_random, y_train_random, ls_flag=True)
    # w_ls, b_ls, = ls(x_train_random, y_train_random)
    y_pred_ls = np.dot(x_test_random, w_ls) + b_ls
    err_ls = rmse_fn(y_test_random, y_pred_ls)[0]
    # TLS
    w_tls, b_tls = lsOrTls_fn(x_train_random, y_train_random, ls_flag=False)
    # w_tls, b_tls = tls(x_train_random, y_train_random)
    y_pred_tls = np.dot(x_test_random, w_tls) + b_tls
    err_tls = rmse_fn(y_test_random, y_pred_tls)[0]
    # EM-TLS
    w_em_list, b_em_list, E, r, target1, target2 = emTest_fn(x_train_random, y_train_random, w_epsilon, correct,
                                                             max_iter_em, plot_stds_wb=plot_stds_wb)
    # 3.1、记录em的迭代
    len_em = len(w_em_list)
    ls_list = [err_ls] * len_em
    tls_list = [err_tls] * len_em
    em_test_list = []
    for em_i in range(len_em):
        y_pred_em = np.dot(x_test_random, w_em_list[em_i]) + b_em_list[em_i]
        em_test_list.append(rmse_fn(y_test_random, y_pred_em)[0])
    # 3.2、绘制 ls_err, tls_err, em_err
    fig, axs = plt.subplots(1, 3, figsize=(12, 6))
    axs[0].plot(ls_list, label='ls', marker='s')
    axs[0].plot(tls_list, label='tls', marker='p')
    axs[0].plot(em_test_list, label='em', marker='*')
    axs[1].plot(target1, label='target1', color='g', marker='*')
    axs[2].plot(target2, label='(now_x + E) @ w_std - now_y - r', color='g', marker='*')
    axs[0].legend()
    axs[1].legend()
    axs[2].legend()
    title = '\ntrain_ratio:' + str(train_ratio) + '  max_iter_em:' + str(max_iter_em) + \
            '  w_epsilon:' + str(w_epsilon) + '  correct:' + str(correct)
    fig.suptitle('seed:' + str(cur_seed) + str(select_feature) + title, fontsize=18)

    # 4、统计结果
    # print("RMSE:======================================")
    # print("ls: ", err_ls)
    # print("tls:", err_tls)
    # print("em: ", em_test_list)
    global count_ls, count_tls, count_em, all_em
    if em_test_list[-1] <= err_tls:
        count_em += 1
        if em_test_list[-1] <= err_ls:
            all_em += 1
        # print("em")
        if save_flag:
            now = datetime.now()
            file_name = now.strftime("%Y%m%d%H%M%S") + str(now.microsecond // 1000).zfill(3) + '.png'
            plt.savefig(os.path.join(file_dir, file_name))
    elif err_tls <= em_test_list[-1]:  # tmp_tls <= tmp_ls and
        count_tls += 1
        # print("tls")
    elif err_ls <= err_tls and err_ls <= em_test_list[-1]:
        count_ls += 1
        # print("ls")

    # 3.3、是否显示：ls-tls-em的结果图、target1:拉格朗日函数、target2:(now_x + E) @ w_std - now_y - r
    if plot_flag:
        plt.show()
    plt.close()

    pass


# pearson:  ['V1/D2/F2', 'D1/F1', 'Area_100_10', 'F6', 'F3', 'F8', 'D3', 'F9', 'F7', 'F4', 'D4', 'D5/F5', 'D6']
# spearman: ['V1/D2/F2', 'Area_100_10', 'D1/F1', 'F8', 'D3', 'D6', 'F9', 'F7', 'F4', 'D4', 'F3', 'D5/F5', 'F6']
if __name__ == '__main__':
    file1, file2 = 'data/dataset.csv', 'data/build_features.csv'
    select_feature1 = ['F2', 'F6', 'F7', 'D6']
    # 20240102: 236(595/1000)， 2368(642/1000)
    select_feature2 = ['V1/D2/F2', 'F6', 'F7', 'D6']

    # 选择文件和特征
    data_all = pd.read_csv(file2)
    select_feature = select_feature2
    data = (data_all[select_feature + ['cycle_life']]).values

    # 初始化参数
    train_ratio = 0.9
    max_iter_em, w_epsilon, correct = 40, 1e-6, 1e-1

    file_dir = 'test_em/236_file2'  # todo:，每次更换特征需要修改
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    count_ls, count_tls, count_em, all_em = 0, 0, 0, 0
    for i in range(0, 1000):
        # print(f"seed {i:03d}====================================================================================")
        test_em(i, plot_flag=False, save_flag=False, plot_stds_wb=False)  # plot_stds_wb是EM内的绘制r_std E_std
    print(select_feature, count_tls, count_em, all_em)

    pass
