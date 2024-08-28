import numpy as np

# 选择聚类方法：clique 类
from pyclustering.cluster.clique import clique
# clique 可视化
from pyclustering.cluster.clique import clique_visualizer

# 构建训练数据
f0 = np.array([37, 42, 49, 56, 61, 65])  # 体重
f1 = np.array([147, 154, 161, 165, 172, 177])  # 身高
f2 = np.array([9, 14, 20, 24, 30, 38])  # 年龄

data = np.array([f0, f1, f2])
data = data.T
data_M = np.array(data)

# 创建 CLIQUE 算法进行处理
# 定义每个维度中网格单元的数量
intervals = 5
# 密度阈值
threshold = 0
clique_instance = clique(data_M, intervals, threshold)

# 开始聚类过程并获得结果
clique_instance.process()
clique_cluster = clique_instance.get_clusters()  # allocated clusters

# 被认为是异常值的点（噪点）
noise = clique_instance.get_noise()
# CLIQUE形成的网格单元
cells = clique_instance.get_cells()

print("Amount of clusters:", len(clique_cluster))
print(clique_cluster)
# 显示由算法形成的网格
clique_visualizer.show_grid(cells, data_M)
# 显示聚类结果
clique_visualizer.show_clusters(data_M, clique_cluster, noise)  # show clustering results
