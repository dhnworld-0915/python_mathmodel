# -*- coding: utf-8 -*-
# ==============================================================================
# 基于sklearn实现DBSCAN算法
# ==============================================================================
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.cluster import dbscan

data, _ = datasets.make_moons(500, noise=0.1, random_state=1)  # 创建数据集
df = pd.DataFrame(data, columns=['feature1', 'feature2'])  # 将数据集转换为dataframe
# 第二步：创建数据集并作可视化处理
# 绘制样本点，s为样本点大小，aplha为透明度，设置图形名称
# 看下面第1个图
df.plot.scatter('feature1', 'feature2', s=100, alpha=0.6,
                title='dataset by make_moon')
plt.show()
# DBSCAN算法
# eps为邻域半径，min_samples为最少点数目
core_samples, cluster_ids = dbscan(data, eps=0.2, min_samples=20)
# cluster_id=k，k为非负整数时，表示对应的点属于第k簇，k为簇的编号，当k=-1时，表示对应的点为噪音点

# np.c_用于合并按行两个矩阵，可以看下面第3个图
# （要求两个矩阵行数相等，这里表示将样本数据特征与对应的簇编号连接）
df = pd.DataFrame(np.c_[data, cluster_ids], columns=['feature1', 'feature2', 'cluster_id'])
# astype函数用于将pandas对象强制转换类型，这里将浮点数转换为整数类型
df['cluster_id'] = df['cluster_id'].astype('int')

# 绘图，c = list(df['cluster_id'])表示样本点颜色按其簇的编号绘制
# cmap=rainbow_r表示颜色从绿到黄，colorbar = False表示删去显示色阶的颜色栏
# 看下面第2个图
df.plot.scatter('feature1', 'feature2', s=100,
                c=list(df['cluster_id']), cmap='rainbow_r', colorbar=False,
                alpha=0.6, title='DBSCAN cluster result')
plt.show()