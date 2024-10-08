# 1.模糊层次聚类
from numpy import array, zeros, triu
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt


def hecheng(a, b):
    m, N = a.shape
    n = b.shape[1]
    c = zeros((m, n))
    for i in range(m):
        for j in range(n):
            c[i, j] = max([min(a[i, k], b[k, j]) for k in range(N)])
    return c


a = array([[5, 5, 3, 2], [2, 3, 4, 5], [5, 5, 2, 3], [1, 5, 3, 1], [2, 4, 5, 1]])
d = array([[sum(abs(a[i] - a[j])) for i in range(5)] for j in range(5)])
r = 1 - 0.1 * d
print(r)
tr = hecheng(r, r)
while abs(r - tr).sum() > 0.00001:
    r = tr
    tr = hecheng(r, r)
print('\n---------------------------\n', tr)
d2 = 1 - tr  # 为了画图，再次转换为距离关系
d2 = triu(d2, 1)
d2 = d2[d2 != 0]  # 提取矩阵上三角中的非零元素
z = sch.linkage(d2)
s = ['I', 'II', 'III', 'IV', 'V']
sch.dendrogram(z, labels=s)  # 画聚类树
plt.yticks([])  # y轴不可见
plt.show()

# 2.模糊C均值聚类
import numpy as np
import matplotlib.pyplot as plt
from skfuzzy.cluster import cmeans

a = np.array([[1, 3], [1.5, 3.2], [1.3, 2.8], [3, 1]])
cntr, u, _, _, _, _, _ = cmeans(a.T, c=2, m=2, error=0.005, maxiter=1000)
# cntr的每一行是一个聚类中心，u的每一列是一个对象的隶属度
print(cntr, '\n-------------------------\n', u)
