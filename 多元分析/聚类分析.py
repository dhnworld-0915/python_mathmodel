# 1.SKlearn中K-均值算法的使用

# KMeans 的主要参数：
# n_clusters: int,default=8　　K值，给定的分类数量，默认值 8。
# init：{‘k-means++’, ‘random’}　　初始中心的选择方式，默认’K-means++'是优化值，也可以随机选择或自行指定。
# n_init：int, default=10　　以不同的中心初值多次运行，以降低初值对算法的影响。默认值 10。
# max_iter：int, default=300　　最大迭代次数。默认值 300。
# algorithm：{“auto”, “full”, “elkan”}, default=”auto”　　算法选择，"full"是经典的 EM算法，“elkan"能快速处理定义良好的簇，默认值 “auto"目前采用"elkan”。

# KMeans 的主要属性：
# **clustercenters：**每个聚类中心的坐标
# labels_： 每个样本的分类结果
# inertia_： 每个点到所属聚类中心的距离之和。

from sklearn.cluster import KMeans  # 导入 sklearn.cluster.KMeans 类
import numpy as np

X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
kmCluster = KMeans(n_clusters=2).fit(X)  # 建立模型并进行聚类，设定 K=2
print(kmCluster.cluster_centers_)  # 返回每个聚类中心的坐标
# [[10., 2.], [ 1., 2.]]  # print 显示聚类中心坐标
print(kmCluster.labels_)  # 返回样本集的分类结果
# [1, 1, 1, 0, 0, 0]  # print 显示分类结果
print(kmCluster.predict([[0, 0], [12, 3]]))  # 根据模型聚类结果进行预测判断
# [1, 0]  # print显示判断结果：样本属于哪个类别

# 2.针对大样本集的改进算法：Mini Batch K-Means

# 对于样本集巨大的问题，例如样本量大于 10万、特征变量大于100，K-Means算法耗费的速度和内存很大。SKlearn 提供了针对大样本集的改进算法 Mini Batch K-Means，并不使用全部样本数据，而是每次抽样选取小样本集进行 K-Means聚类，进行循环迭代。Mini Batch K-Means 虽然性能略有降低，但极大的提高了运行速度和内存占用。

# MiniBatchKMeans 与 KMeans不同的主要参数是：
# batch_size: int, default=100　　 抽样集的大小。默认值 100。

from sklearn.cluster import MiniBatchKMeans  # 导入 .MiniBatchKMeans 类
import numpy as np

X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 0], [4, 4],
              [4, 5], [0, 1], [2, 2], [3, 2], [5, 5], [1, -1]])
# fit on the whole data&result
mbkmCluster = MiniBatchKMeans(n_clusters=2, batch_size=6, max_iter=10).fit(X)  # 抽样集大小为100
print(mbkmCluster.cluster_centers_)  # 返回每个聚类中心的坐标
# [[3.96,2.41], [1.12,1.39]] # print 显示内容
print(mbkmCluster.labels_)  # 返回样本集的分类结果
# [1 1 1 0 0 0 0 1 1 0 0 1]  # print 显示内容
print(mbkmCluster.predict([[0, 0], [4, 5]]))  # 根据模型聚类结果进行预测判断
# [1, 0]  # 显示判断结果：样本属于哪个类别

# 3.K-Means cluster by scikit-learn for problem "education2015"
#   K-Means 聚类算法（SKlearn）求解：各地区高等教育发展状况-2015 问题

#  -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans


def main():
    # 读取数据文件
    readPath = "data&result/education2015.xlsx"  # 数据文件的地址和文件名
    dfFile = pd.read_excel(readPath, header=0)  # 首行为标题行
    dfFile = dfFile.dropna()  # 删除含有缺失值的数据
    # print(dfFile.dtypes)  # 查看 df 各列的数据类型
    # print(dfFile.shape)  # 查看 df 的行数和列数
    print(dfFile.head())

    # 数据准备
    z_scaler = lambda x: (x - np.mean(x)) / np.std(x)  # 定义数据标准化函数
    dfScaler = dfFile[['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10']].apply(z_scaler)  # 数据归一化
    dfData = pd.concat([dfFile[['地区']], dfScaler], axis=1)  # 列级别合并
    df = dfData.loc[:, ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10']]  # 基于全部 10个特征聚类分析
    # df = dfData.loc[:,['x1','x2','x7','x8','x9','x10']]  # 降维后选取 6个特征聚类分析
    X = np.array(df)  # 准备 sklearn.cluster.KMeans 模型数据
    print("Shape of cluster data&result:", X.shape)

    # KMeans 聚类分析(sklearn.cluster.KMeans)
    nCluster = 4
    kmCluster = KMeans(n_clusters=nCluster).fit(X)  # 建立模型并进行聚类，设定 K=2
    print("Cluster centers:\n", kmCluster.cluster_centers_)  # 返回每个聚类中心的坐标
    print("Cluster results:\n", kmCluster.labels_)  # 返回样本集的分类结果

    # 整理聚类结果
    listName = dfData['地区'].tolist()  # 将 dfData 的首列 '地区' 转换为 listName
    dictCluster = dict(zip(listName, kmCluster.labels_))  # 将 listName 与聚类结果关联，组成字典
    listCluster = [[] for k in range(nCluster)]
    for v in range(0, len(dictCluster)):
        k = list(dictCluster.values())[v]  # 第v个城市的分类是 k
        listCluster[k].append(list(dictCluster.keys())[v])  # 将第v个城市添加到 第k类
    print("\n聚类分析结果(分为{}类):".format(nCluster))  # 返回样本集的分类结果
    for k in range(nCluster):
        print("第 {} 类：{}".format(k, listCluster[k]))  # 显示第 k 类的结果

    return


if __name__ == '__main__':
    main()
