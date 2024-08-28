# 导入相关库
import copy
import pandas as pd
import numpy as np

# 读取数据
data = pd.read_excel("data&result/shangquan.xlsx")
print(data)

# 提取变量名
label_need = data.keys()[2:]
print(label_need)
data1 = data[label_need].values  # 只取指标数据

# 指标正向化处理后数据为data2
data2 = data1
print(data2)

# 越小越优指标位置,注意python是从0开始计数，对应位置也要相应减1
index = [2, 3]
for i in range(0, len(index)):
    data2[:, index[i]] = max(data1[:, index[i]]) - data1[:, index[i]]
print(data2)

# 某点最优型指标
index1 = [4]
a = 90  # 最优型数值
for i in range(0, len(index1)):
    data2[:, index1[i]] = 1 - abs(data1[:, index1[i]] - a) / max(abs(data1[:, index1[i]] - a))
print(data2)

# 0.002~1区间归一化
[m, n] = data2.shape
data3 = copy.deepcopy(data2)
ymin = 0.002
ymax = 1
for j in range(0, n):
    d_max = max(data2[:, j])
    d_min = min(data2[:, j])
    data3[:, j] = (ymax - ymin) * (data2[:, j] - d_min) / (d_max - d_min) + ymin
print(data3)

# 计算信息熵
p = copy.deepcopy(data3)
for j in range(0, n):
    p[:, j] = data3[:, j] / sum(data3[:, j])
print(p)
E = copy.deepcopy(data3[0, :])
for j in range(0, n):
    E[j] = -1 / np.log(m) * sum(p[:, j] * np.log(p[:, j]))
print(E)

# 计算权重
w = (1 - E) / sum(1 - E)
print(w)

# 计算得分
s = np.dot(data3, w)
Score = 100 * s / max(s)
for i in range(0, len(Score)):
    print(f"第{i}个评价对象得分为：{Score[i]}")
