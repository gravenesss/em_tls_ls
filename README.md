

# EM-TLS的锂电池寿命预测

## 1. 项目简介
0. 首先需要将数据集中的数据转为pkl格式，运行BuildPkl_Batch1.py、BuildPkl_Batch2.py、BuildPkl_Batch3.py即可。
1. 将pkl文件中的数据提取原论文的特征以及Area特征到 data/build_features.csv 文件中。
2. test1_em.py 主要是用来选取特征的，随机划分数据集1000次，来查看效果(1000次内 tls<em的次数；em<=tls的次数；em<=tls且em<=ls的次数)
3. test2_noise.py 是噪声比例依次增大的实验，超参数包括：训练集比例；EM中矫正向量、EM中迭代次数；数据集划分次数、随机划分噪声次数、噪声模式。
4. test3_train.py 是训练集比例依次增大的实验。





## 2. 细节介绍

test1_em2.py、test2_noise.py、test3_train.py 直接调用 ls、tls、em算法(均在 now_utils.py中，ls和tls简化为了lsOrTls_fn函数)进行实验即可。

其中 ls/tls 的步骤包括：①标准化；②计算w_std；③还原为w, b。

其中 em  的步骤包括：①标准化，②初始化 diag_x, 使用 tlsStd_fn 函数初始化 w_std；③迭代：E步：通过 diag_x 和 w_std 计算 E 和 r 并更新 diag_x；M步：使用 tlsStd_fn 函数更新 w_std；④还原为w, b。