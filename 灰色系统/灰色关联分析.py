"""适用于对于两个系统之间的因素，其随时间或不同对象而变化的关联性大小的量度，称为关联度。
在系统发展过程中，若两个因素变化的趋势具有一致性，即同步变化程度较高，即可谓二者关联程度较高；反之，则较低。
因此，灰色关联分析方法，是根据因素之间发展趋势的相似或相异程度，亦即“灰色关联度”，作为衡量因素间关联程度的一种方法。"""

# 导入可能要用到的库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 读取数据
wine = pd.read_excel("data&result/wine.xls", index_col=0)
print(wine)


# 无量纲化
def dimensionlessProcessing(df_values, df_columns):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    res = scaler.fit_transform(df_values)
    return pd.DataFrame(res, columns=df_columns)


# 求第一列(影响因素)和其它所有列(影响因素)的灰色关联值
def GRA_ONE(data, m=0):  # m为参考列
    # 标准化
    data = dimensionlessProcessing(data.values, data.columns)
    # 参考数列
    std = data.iloc[:, m]
    # 比较数列
    ce = data.copy()

    n = ce.shape[0]
    m = ce.shape[1]

    # 与参考数列比较，相减
    grap = np.zeros([n, m])
    for i in range(m):
        for j in range(n):
            grap[j, i] = abs(ce.iloc[j, i] - std[j])

    # 取出矩阵中的最大值和最小值
    mmax = np.amax(grap)
    mmin = np.amin(grap)
    ρ = 0.5  # 灰色分辨系数

    # 计算值
    grap = pd.DataFrame(grap).map(lambda x: (mmin + ρ * mmax) / (x + ρ * mmax))

    # 求均值，得到灰色关联值
    RT = grap.mean(axis=0)
    return pd.Series(RT)


# 调用GRA_ONE，求得所有因素之间的灰色关联值
def GRA(data):
    list_columns = np.arange(data.shape[1])
    df_local = pd.DataFrame(columns=list_columns)
    for i in np.arange(data.shape[1]):
        df_local.iloc[:, i] = GRA_ONE(data, m=i)
    return df_local


data_gra = GRA(wine)
print(data_gra)
# 结果可视化
import seaborn as sns  # 可视化图形调用库
import matplotlib.pyplot as plt


def ShowGRAHeatMap(data):
    # 色彩集
    colormap = plt.cm.RdBu
    plt.figure(figsize=(18, 16))
    plt.title('Person Correlation of Features', y=1.05, size=18)
    sns.heatmap(data.astype(float), linewidths=0.1, vmax=1.0, square=True,
                cmap=colormap, linecolor='white', annot=True)
    plt.show()


ShowGRAHeatMap(data_gra)
