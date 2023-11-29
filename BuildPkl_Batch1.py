import h5py
import matplotlib.pyplot as plt
import numpy as np
import pickle

# import scipy.io

base_dir = '/data/'
matFilename = base_dir + '2017-05-12_batchdata_updated_struct_errorcorrect.mat'

# 打开MATLAB数据文件并存储在变量中
f = h5py.File(matFilename)
# 打印 HDF5 文件中的键（数据集）列表。  ['#refs#', '#subsystem#', 'batch', 'batch_date']
print(list(f.keys()))
# 带有 HDF5 文件中的键“batch”的数据集存储在变量中。
batch = f['batch']
# 数据集中的键列表/属性。 ['Vdlin', 'barcode','channel_id','cycle_life','cycles','policy','policy_readable','summary']
print(list(batch.keys()))

# 提取单元格数并将其存储在  46个
num_cells = batch['summary'].shape[0]
bat_dict = {}
# 汇总数据包括每个周期的信息，包括循环次数cycle、放电容量QDischarge、充电容量QCharge、内阻IR、最高温度Tmax、平均温度Tavg、最低温度Tmin和充电时间chargetime。
# 循环数据包括一个周期内的信息，包括时间t、充电容量Qc、放电容量Qd、电流I、电压V、温度T。我们还包括放电discharge_dQdV，线性插值放电容量(Qdlin)和线性插值温度(Tdlin)的派生向量。
for i in range(num_cells):  # 第 i 行数据
    # 获取当前单元格的循环寿命
    cl = f[batch['cycle_life'][i, 0]].value
    # 提取当前单元格的“policy_readable”，将其转换为字节，然后将其解码为字符串 使用的都是policy_readable
    policy = f[batch['policy_readable'][i, 0]].value.tobytes()[::2].decode()
    # 提取 通道id
    channel_id = f[batch['channel_id'][i, 0]].value
    # 提取 vdlin
    vdlin = f[batch['Vdlin'][i, 0]].value

    # 各种行提取和处理当前单元格的摘要数据，包括“IR”、“QC”、“QD”、“TAVG”、“Tmin”、“Tmax”、“充电时间”和“周期”。处理后放到 summary
    # np.hstack用于沿水平方向将多个数组堆叠在一起
    summary_IR = np.hstack(f[batch['summary'][i, 0]]['IR'][0, :].tolist())
    summary_QC = np.hstack(f[batch['summary'][i, 0]]['QCharge'][0, :].tolist())
    summary_QD = np.hstack(f[batch['summary'][i, 0]]['QDischarge'][0, :].tolist())
    summary_TA = np.hstack(f[batch['summary'][i, 0]]['Tavg'][0, :].tolist())
    summary_TM = np.hstack(f[batch['summary'][i, 0]]['Tmin'][0, :].tolist())
    summary_TX = np.hstack(f[batch['summary'][i, 0]]['Tmax'][0, :].tolist())
    summary_CT = np.hstack(f[batch['summary'][i, 0]]['chargetime'][0, :].tolist())
    summary_CY = np.hstack(f[batch['summary'][i, 0]]['cycle'][0, :].tolist())
    summary = {'IR': summary_IR, 'QC': summary_QC, 'QD': summary_QD, 'Tavg': summary_TA, 'Tmin': summary_TM,
               'Tmax': summary_TX, 'chargetime': summary_CT, 'cycle': summary_CY}

    # 获取当前单元格的循环轮数数据
    cycles = f[batch['cycles'][i, 0]]
    # 遍历每个循环的数据，处理并存储到cycle_dict中
    cycle_dict = {}
    for j in range(cycles['I'].shape[0]):
        I = np.hstack(f[cycles['I'][j, 0]].value)  # 可以只使用一个括号
        Qc = np.hstack(f[cycles['Qc'][j, 0]].value)
        Qd = np.hstack(f[cycles['Qd'][j, 0]].value)
        Qdlin = np.hstack(f[cycles['Qdlin'][j, 0]].value)
        T = np.hstack(f[cycles['T'][j, 0]].value)
        Tdlin = np.hstack(f[cycles['Tdlin'][j, 0]].value)
        V = np.hstack(f[cycles['V'][j, 0]].value)
        dQdV = np.hstack(f[cycles['discharge_dQdV'][j, 0]].value)
        t = np.hstack(f[cycles['t'][j, 0]].value)
        cd = {'I': I, 'Qc': Qc, 'Qd': Qd, 'Qdlin': Qdlin, 'T': T, 'Tdlin': Tdlin, 'V': V, 'dQdV': dQdV, 't': t}
        cycle_dict[str(j)] = cd

    # 将循环数据和摘要数据存储到cell_dict中
    cell_dict = {'cycle_life': cl, 'charge_policy': policy, 'channel_id': channel_id, 'vdlin': vdlin,
                 'summary': summary, 'cycles': cycle_dict}
    # 使用单元格索引作为键，将cell_dict存储到bat_dict中
    key = 'b1c' + str(i)
    bat_dict[key] = cell_dict
print(bat_dict.keys())  # 46个 dict_keys(['b1c0', ..., 'b1c45'])

# 循环轮数 和 放电容量的关系
plt.plot(bat_dict['b1c43']['summary']['cycle'], bat_dict['b1c43']['summary']['QD'])
plt.show()
# 第10个循环的放电容量 和 电压的关系。TODO：电压骤变是什么含义？？
plt.plot(bat_dict['b1c43']['cycles']['10']['Qd'], bat_dict['b1c43']['cycles']['10']['V'])
plt.show()

# 将处理后的数据保存到磁盘上的pickle文件中：
with open(base_dir + 'batch1.pkl', 'wb') as fp:
    pickle.dump(bat_dict, fp)

# 三组数据分别：46 48 46 → Data_load中进行筛出。  → 41 43 40
