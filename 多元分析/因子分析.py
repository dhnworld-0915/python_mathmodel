import pandas as pd
from factor_analyzer import FactorAnalyzer
import matplotlib.pyplot as plt
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 指定默认字体：解决plot不能显示中文问题
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

# 示例数据集
data = pd.read_excel('data&result/data.xlsx', usecols='B:K')  # 请替换为您自己的数据集

# 创建因子分析对象，指定因子数量
fa = FactorAnalyzer(n_factors=3, rotation='varimax')

# 执行因子分析
fa.fit(data)

# 提取因子载荷矩阵
loadings = fa.loadings_

# 输出因子载荷矩阵
print("因子载荷矩阵：")
print(loadings)

# 绘制因子载荷图
plt.figure(figsize=(8, 6))
plt.imshow(loadings, cmap='coolwarm', interpolation='nearest')
plt.colorbar()
plt.title("因子载荷图")
plt.xticks(range(data.shape[1]), data.columns, rotation=90)
plt.yticks(range(fa.n_factors), [f"Factor {i + 1}" for i in range(fa.n_factors)])
plt.show()
