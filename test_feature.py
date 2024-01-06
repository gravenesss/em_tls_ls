import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

file1, file2 = 'data/dataset.csv', 'data/build_features.csv'  # 'D1/F1', 'D2/F2', 'D5/F5', 'V1/D2/F2'
select_feature1 = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9',  'D3', 'D4',  'D6']
select_feature2 = ['V1/D2/F2', 'D1/F1', 'Area_100_10', 'F6', 'F3', 'F8', 'D3', 'F9', 'F7', 'F4', 'D4', 'D5/F5', 'D6']  # 20240102: 236(595/1000)， 2368(642/1000)

# 5:['F3' 'F4' 'F5' 'F8' 'F9']                4:['F3' 'F5' 'F8' 'F9']             3:['F3' 'F8' 'F9']
# 6:['F1' 'F3' 'F4' 'F5' 'F8' 'F9']  5: ['Area_100_10' 'F3' 'F8' 'F9' 'D5/F5']   4:['Area_100_10' 'F3' 'F8' 'F9']    3:['Area_100_10' 'F3' 'F9']

# 选择文件和特征
data_all = pd.read_csv(file2)
select_feature = select_feature2
data = (data_all[select_feature + ['cycle_life']]).values
X = data[:, :-1]
y = data[:, -1]
# 设置要保留的特征数量（k）
k = 6

# 使用RFE进行特征选择
model = LinearRegression()
# 使用随机森林模型
# model = RandomForestRegressor()
rfe = RFE(model, n_features_to_select=k)
X_selected = rfe.fit_transform(X, y)

# 获取被选择的特征的索引
selected_feature_indices = np.where(rfe.support_)[0]

# 使用NumPy数组或Pandas Series来获取被选择的特征
selected_features = np.array(select_feature)[selected_feature_indices]

# 或者，如果你想要一个列表
# selected_features = list(np.array(select_feature)[selected_feature_indices])

print("Selected features:", selected_features)
