import numpy as np


data_size = 124

for split in range(10):
    np.random.seed(split)
    random_indices = np.random.permutation(data_size)  # 随机排序
    print(random_indices)

print("end=================")
for p in range(10):
    # 划分训练集与测试集
    np.random.seed(p)  # 保证每个训练集比例所对应的s次随机排列的顺序一致，每次排序都要使用，写在循环外不起作用
    random_datax = np.random.permutation(np.arange(0, 124, 1))  # 随机排序
    print(random_datax)
