import numpy as np
import pandas as pd
import pickle
from scipy import integrate
from os.path import join

base_dir = '/data'  # .pkl files
DATA_DIR = '/data'  # features extraction


# 把数据加载为字典：过程中对基本异常数据进行处理。
def load_batches_to_dict(amount_to_load=3):
    if amount_to_load < 1 or amount_to_load > 3:
        raise "amount_to_load is not a valid number! Try a number between 1 and 3."

    batches_dict = {}  # Initializing

    # 1. 加载 batch1
    print("Loading batch1 ...")
    path1 = join(base_dir, "batch1.pkl")  # .\batch1.pkl   ../Data\batch1.pkl
    batch1 = pickle.load(open(path1, 'rb'))
    # 移除未达到80%容量的电池
    del batch1['b1c8']
    del batch1['b1c10']
    del batch1['b1c12']
    del batch1['b1c13']
    del batch1['b1c22']
    # updates/replaces 将 batch1 字典的键值对更新到 batches_dict
    batches_dict.update(batch1)  # 原来：bat_dict = {**batch1, **batch2, **batch3}

    # 2. 更新 batch1 加入 batch2
    if amount_to_load > 1:
        print("Loading batch2 ...")
        path2 = join(base_dir, "batch2.pkl")
        batch2 = pickle.load(open(path2, 'rb'))

        # batch1中有 5个电池 被带入batch2，我们将从batch2中删除数据，
        # 并将其与batch1的正确单元格放在一起 → 总的：48-5=43
        batch2_keys = ['b2c7', 'b2c8', 'b2c9', 'b2c15', 'b2c16']
        batch1_keys = ['b1c0', 'b1c1', 'b1c2', 'b1c3', 'b1c4']
        add_len = [662, 981, 1060, 208, 482]

        for i, bk in enumerate(batch1_keys):
            batch1[bk]['cycle_life'] = batch1[bk]['cycle_life'] + add_len[i]
            for j in batch1[bk]['summary'].keys():
                if j == 'cycle':
                    batch1[bk]['summary'][j] = np.hstack((batch1[bk]['summary'][j],
                                                          batch2[batch2_keys[i]]['summary'][j] + len(
                                                              batch1[bk]['summary'][j])))
                else:
                    batch1[bk]['summary'][j] = np.hstack(
                        (batch1[bk]['summary'][j], batch2[batch2_keys[i]]['summary'][j]))
            last_cycle = len(batch1[bk]['cycles'].keys())
            # 第一批电池循环数据的键。遍历第二批电池数据中对应索引的键和值，将其添加到第一批电池数据的循环字典中。
            for j, jk in enumerate(batch2[batch2_keys[i]]['cycles'].keys()):
                batch1[bk]['cycles'][str(last_cycle + j)] = batch2[batch2_keys[i]]['cycles'][jk]

        del batch2['b2c7']
        del batch2['b2c8']
        del batch2['b2c9']
        del batch2['b2c15']
        del batch2['b2c16']

        # All keys have to be updated after the reordering.
        batches_dict.update(batch1)
        batches_dict.update(batch2)

    # 3. 加入 batch3
    if amount_to_load > 2:
        print("Loading batch3 ...")
        path3 = join(base_dir, "batch3.pkl")
        batch3 = pickle.load(open(path3, 'rb'))

        # remove noisy channels from batch3
        del batch3['b3c37']
        del batch3['b3c2']
        del batch3['b3c23']
        del batch3['b3c32']
        del batch3['b3c38']
        del batch3['b3c39']

        batches_dict.update(batch3)

    print("Done loading batches")
    return batches_dict


# 构建特征：主要问题：取对数时使用的底数； 下标，什么时候算第100个循环。
def build_feature_df(batch_dict):
    """ 返回一个 pandas DataFrame，其中包含加载的批处理字典中所有最初使用的功能"""
    from scipy.stats import skew, kurtosis
    from sklearn.linear_model import LinearRegression

    print("Start building features ...")
    n_cells = len(batch_dict.keys())  # 124 cells (3 batches)

    # Initializing feature vectors:
    if True:
        cycle_life = np.zeros(n_cells)
        # 1. delta_Q_100_10(V)  (ΔQ100-10(V))中提取特征
        minimum_dQ_100_10 = np.zeros(n_cells)
        variance_dQ_100_10 = np.zeros(n_cells)
        skewness_dQ_100_10 = np.zeros(n_cells)
        kurtosis_dQ_100_10 = np.zeros(n_cells)

        # 2. Discharge capacity fade curve features 放电容量衰退曲线特征。
        slope_lin_fit_2_100 = np.zeros(n_cells)         # 容量衰减曲线的线性拟合斜率Slope，周期 2 至 100
        intercept_lin_fit_2_100 = np.zeros(n_cells)     # 容量面曲线线性拟合的截距Intercept，周期 2 至 100
        discharge_capacity_2 = np.zeros(n_cells)        # Discharge capacity, cycle 2
        diff_discharge_capacity_max_2 = np.zeros(n_cells)  # 最大放电容量与循环2之间的差异

        # 3. Other features
        mean_charge_time_2_6 = np.zeros(n_cells)        # 平均充电时间，周期 2 至 6  F6
        temp_integral_2_100 = np.zeros(n_cells)         # 2到100次循环的 温度积分  F7
        minimum_IR_2_100 = np.zeros(n_cells)            # 最小内阻 F8
        diff_IR_100_2 = np.zeros(n_cells)               # 第100次循环与第2次循环之间内阻的差异 F9

        # 4. My features
        qd_v_area_100_10 = np.zeros(n_cells)           # 100-10循环的放电容量电压面积 Area_100_10
        pass

    # iterate/loop over all cells. 迭代/循环所有单元格。
    for i, cell in enumerate(batch_dict.values()):
        cycle_life[i] = cell['cycle_life']
        # 1. delta_Q_100_10(V)
        c10 = cell['cycles']['10']  # 这个直接是属性值，不是下标。
        c100 = cell['cycles']['100']
        dQ_100_10 = c100['Qdlin'] - c10['Qdlin']
        minimum_dQ_100_10[i] = np.log10(np.abs(np.min(dQ_100_10)))    # 都取了log10
        variance_dQ_100_10[i] = np.log10(np.abs(np.var(dQ_100_10)))   # 可以不需要abs 本身就是正的
        skewness_dQ_100_10[i] = np.log10(np.abs(skew(dQ_100_10)))
        kurtosis_dQ_100_10[i] = np.log10(np.abs(kurtosis(dQ_100_10)))

        # 2. Discharge capacity fade curve features  下标是1~99，左闭右开
        q = cell['summary']['QD'][1:100].reshape(-1, 1)     # 放电容量; q.shape = (99, 1);
        X = cell['summary']['cycle'][1:100].reshape(-1, 1)  # 循环数从 2 到 100; X.shape = (99, 1)
        # 获得 斜率和截距；x
        linear_regressor_2_100 = LinearRegression()
        linear_regressor_2_100.fit(X, q)
        slope_lin_fit_2_100[i] = linear_regressor_2_100.coef_[0]
        intercept_lin_fit_2_100[i] = linear_regressor_2_100.intercept_
        discharge_capacity_2[i] = q[0][0]
        diff_discharge_capacity_max_2[i] = np.max(q) - q[0][0]

        # 3. Other features： 1 2 3 4 5 的下标。对应2-6的充电时间取平均值
        mean_charge_time_2_6[i] = np.mean(cell['summary']['chargetime'][1:6])
        temp_integral_2_100[i] = np.log10(np.abs(integrate.simps(cell['summary']['Tavg'][1:100])))  # 可以考虑取对数
        # min_ir_ = min(ir[ir > 0])  #  内阻没有负数。
        minimum_IR_2_100[i] = np.min(cell['summary']['IR'][1:100])
        diff_IR_100_2[i] = cell['summary']['IR'][99] - cell['summary']['IR'][1]

        # 4.
        v = cell['vdlin'][0]  # (1000,)  不取[0]报错
        # print(dQ_100_10.shape, v.shape)
        # qd_v_area_100_10[i] = np.abs(np.sum((v[1:] - v[:-1]) * dQ_100_10[1:]))
        qd_v_area_100_10[i] = np.abs(integrate.simps(dQ_100_10, v))

    # 将所有特征组合在一个大矩阵中，其中行是电池，列是特征；最后两个变量是机器学习的标签，即循环寿命和cycle_550_clf
    # (原论文12个，去掉了循环2的内阻、周期2~100温度随时间的积分， F7、 F8
    feature_df = pd.DataFrame({
        "cell_key": np.array(list(batch_dict.keys())),
        "D1/F1": minimum_dQ_100_10,
        "V1/D2/F2": variance_dQ_100_10,
        "D3": skewness_dQ_100_10,
        "D4": kurtosis_dQ_100_10,
        "F3": slope_lin_fit_2_100,
        "F4": intercept_lin_fit_2_100,
        "D5/F5": discharge_capacity_2,
        "D6": diff_discharge_capacity_max_2,

        "F6": mean_charge_time_2_6,
        "F7": temp_integral_2_100,
        "F8": minimum_IR_2_100,
        "F9": diff_IR_100_2,

        "Area_100_10": qd_v_area_100_10,

        "cycle_life": cycle_life,
    })

    print("Done building features")
    return feature_df


if __name__ == "__main__":
    all_batches_dict = load_batches_to_dict()
    features_df = build_feature_df(all_batches_dict)

    save_csv_path = DATA_DIR + '/build_features.csv'
    features_df.to_csv(save_csv_path, index=False)
    print("Saved features to ", save_csv_path)
