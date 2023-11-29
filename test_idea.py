import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 创建一个包含相关数据的DataFrame（示例数据）
data = pd.DataFrame({'A': [1, 2, 3, 4, 5],
                     'B': [5, 4, 3, 2, 1],
                     'C': [2, 4, 6, 8, 10],
                     'D': [10, 8, 6, 4, 2]})

# 计算Pearson相关系数
pearson_corr = data.corr(method='pearson')

# 计算Spearman相关系数
spearman_corr = data.corr(method='spearman')

# 绘制Pearson相关系数热力图
plt.figure(figsize=(8, 6))
sns.heatmap(pearson_corr, annot=True, cmap='coolwarm')
plt.title('Pearson Correlation Heatmap')
plt.show()

# 绘制Spearman相关系数热力图
plt.figure(figsize=(8, 6))
sns.heatmap(spearman_corr, annot=True, cmap='coolwarm')
plt.title('Spearman Correlation Heatmap')
plt.show()

