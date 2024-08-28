# 1.一元线性回归

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std


# 主程序
def main():  # 主程序

    # 生成测试数据:
    nSample = 100
    x1 = np.linspace(0, 10, nSample)  # 起点为 0，终点为 10，均分为 nSample个点
    e = np.random.normal(size=len(x1))  # 正态分布随机数
    yTrue = 2.36 + 1.58 * x1  # y = b0 + b1*x1
    yTest = yTrue + e  # 产生模型数据

    # 一元线性回归：最小二乘法(OLS)
    X = sm.add_constant(x1)  # 向矩阵 X 添加截距列（x0=[1,...1]）
    model = sm.OLS(yTest, X)  # 建立最小二乘模型（OLS）
    results = model.fit()  # 返回模型拟合结果
    yFit = results.fittedvalues  # 模型拟合的 y值
    prstd, ivLow, ivUp = wls_prediction_std(results)  # 返回标准偏差和置信区间

    # OLS model: Y = b0 + b1*X + e
    print(results.summary())  # 输出回归分析的摘要
    print("\nOLS model: Y = b0 + b1 * x")  # b0: 回归直线的截距，b1: 回归直线的斜率
    print('Parameters: ', results.params)  # 输出：拟合模型的系数

    # 绘图：原始数据点，拟合曲线，置信区间
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(x1, yTest, 'o', label="data&result")  # 原始数据
    ax.plot(x1, yFit, 'r-', label="OLS")  # 拟合数据
    ax.plot(x1, ivUp, '--', color='orange', label="upConf")  # 95% 置信区间 上限
    ax.plot(x1, ivLow, '--', color='orange', label="lowConf")  # 95% 置信区间 下限
    ax.legend(loc='best')  # 显示图例
    plt.title('OLS linear regression ')
    plt.show()
    return


if __name__ == '__main__':
    main()

# 2.多元线性回归

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std


# 主程序
def main():  # 主程序

    # 生成测试数据:
    nSample = 100
    x0 = np.ones(nSample)  # 截距列 x0=[1,...1]
    x1 = np.linspace(0, 20, nSample)  # 起点为 0，终点为 10，均分为 nSample个点
    x2 = np.sin(x1)
    x3 = (x1 - 5) ** 2
    X = np.column_stack((x0, x1, x2, x3))  # (nSample,4): [x0,x1,x2,...,xm]
    beta = [5., 0.5, 0.5, -0.02]  # beta = [b1,b2,...,bm]
    yTrue = np.dot(X, beta)  # 向量点积 y = b1*x1 + ...+ bm*xm
    yTest = yTrue + 0.5 * np.random.normal(size=nSample)  # 产生模型数据

    # 多元线性回归：最小二乘法(OLS)
    model = sm.OLS(yTest, X)  # 建立 OLS 模型: Y = b0 + b1*X + ... + bm*Xm + e
    results = model.fit()  # 返回模型拟合结果
    yFit = results.fittedvalues  # 模型拟合的 y值
    print(results.summary())  # 输出回归分析的摘要
    print("\nOLS model: Y = b0 + b1*X + ... + bm*Xm")
    print('Parameters: ', results.params)  # 输出：拟合模型的系数

    # 绘图：原始数据点，拟合曲线，置信区间
    prstd, ivLow, ivUp = wls_prediction_std(results)  # 返回标准偏差和置信区间
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(x1, yTest, 'o', label="data&result")  # 实验数据（原始数据+误差）
    ax.plot(x1, yTrue, 'b-', label="True")  # 原始数据
    ax.plot(x1, yFit, 'r-', label="OLS")  # 拟合数据
    ax.plot(x1, ivUp, '--', color='orange', label="ConfInt")  # 置信区间 上届
    ax.plot(x1, ivLow, '--', color='orange')  # 置信区间 下届
    ax.legend(loc='best')  # 显示图例
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    return


if __name__ == '__main__':
    main()

#     Dep.Variable: y 因变量
#     Model：OLS 最小二乘模型
#     Method: Least Squares 最小二乘
#     No. Observations: 样本数据的数量
#     Df Residuals：残差自由度(degree of freedom of residuals)
#     Df Model：模型自由度(degree of freedom of model)
#     Covariance Type：nonrobust 协方差阵的稳健性
#     R-squared：R 判定系数
#     Adj. R-squared: 修正的判定系数
#     F-statistic： 统计检验 F 统计量
#     Prob (F-statistic): F检验的 P值
#     Log likelihood: 对数似然
#
#     coef：自变量和常数项的系数，b1,b2,...bm,b0
#     std err：系数估计的标准误差
#     t：统计检验 t 统计量
#     P>|t|：t 检验的 P值
#     [0.025, 0.975]：估计参数的 95%置信区间的下限和上限
#     Omnibus：基于峰度和偏度进行数据正态性的检验
#     Prob(Omnibus)：基于峰度和偏度进行数据正态性的检验概率
#     Durbin-Watson：检验残差中是否存在自相关
#     Skewness：偏度，反映数据分布的非对称程度
#     Kurtosis：峰度，反映数据分布陡峭或平滑程度
#     Jarque-Bera(JB)：基于峰度和偏度对数据正态性的检验
#     Prob(JB)：Jarque-Bera(JB)检验的 P值。
#     Cond. No.：检验变量之间是否存在精确相关关系或高度相关关系。

# 3.模型数据的准备

import pandas as pd

"""
# 3.1读取数据文件

df = pd.read_csv("./example.csv", engine="python", encoding="utf_8_sig")
# engine="python"允许处理中文路径，encoding="utf_8_sig"允许读取中文数据

df = pd.read_excel("./example.xls", sheetname='Sheet1', header=0, encoding="utf_8_sig")
# sheetname 表示读取的sheet，header=0 表示首行为标题行， encoding 表示编码方式

df = pd.read_table("./example.txt", sep="\t", header=None)
# sep 表示分隔符，header=None表示无标题行，第一行是数据

# 3.2数据的拆分与合并
# 3.2.1拆分
# 将 Excel 文件分割为多个文件
dfData = pd.read_excel('./example.xls', sheetname='Sheet1')
nRow, nCol = dfData.shape  # 获取数据的行列
# 假设数据共有198,000行，分割为 20个文件，每个文件 10,000行
for i in range(0, int(nRow / 10000) + 1):
    saveData = dfData.iloc[i * 10000 + 1:(i + 1) * 10000 + 1, :]  # 每隔 10,000
    fileName = './example_{}.xls'.format(str(i))
    saveData.to_excel(fileName, sheet_name='Sheet1', index=False)
# 3.2.2合并
# 将多个 Excel 文件合并为一个文件
# 两个 Excel 文件合并
data1 = pd.read_excel('./example0.xls', sheetname='Sheet1')
data2 = pd.read_excel('./example1.xls', sheetname='Sheet1')
data&result = pd.concat([data1, data2])
# 多个 Excel 文件合并
dfData = pd.read_excel('./example0.xls', sheetname='Sheet1')
for i in range(1, 20):
    fileName = './example_{}.xls'.format(str(i))
    dfNew = pd.read_excel(fileName)
    dfData = pd.concat([dfData, dfNew])
dfData.to_excel('./example', index = False)
"""

# 3.3数据的预处理
'''
# 3.3.1缺失数据的处理
dfNew = dfData.dropna(axis=0))  # 删除含有缺失值的行
# 3.3.2重复数据的处理
dfNew = dfData.drop_duplicates(inplace=True)  # 删除重复的数据行
# 3.3.3异常值处理
dfData.boxplot()  # 绘制箱形图
# 按行删除，drop() 默认 axis=0 按行删除
dfNew = dfData.drop(labels=0)  # 按照行号 labels，删除 行号为 0 的行
dfNew = dfData.drop(index=dfData[dfData['A'] == -1].index[0])  # 按照条件检索，删除 dfData['A']=-1 的行
'''

# 4.可视化
"""数据文件中收集了 30个月本公司牙膏销售量、价格、广告费用及同期的市场均价。
　　（1）分析牙膏销售量与价格、广告投入之间的关系，建立数学模型；
　　（2）估计所建立数学模型的参数，进行统计分析；
　　（3）利用拟合模型，预测在不同价格和广告费用下的牙膏销售量。"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    # 读取数据文件
    readPath = "F:/Python Project/Mathematical modeling/data&result/toothpaste.csv"  # 数据文件的地址和文件名
    dfOpenFile = pd.read_csv(readPath, header=0, sep=",")  # 间隔符为逗号，首行为标题行

    # 准备建模数据：分析因变量 Y(sales) 与 自变量 x1~x4  的关系
    dfData = dfOpenFile.dropna()  # 删除含有缺失值的数据

    sns.set_style('dark')
    # 数据探索：分布特征
    fig1, axes = plt.subplots(2, 2, figsize=(10, 8))  # 创建一个 2行 2列的画布
    sns.histplot(dfData['price'], bins=10, ax=axes[0, 0])  # axes[0,0] 左上图
    sns.histplot(dfData['average'], bins=10, ax=axes[0, 1])  # axes[0,1] 右上图
    sns.histplot(dfData['advertise'], bins=10, ax=axes[1, 0])  # axes[1,0] 左下图
    sns.histplot(dfData['difference'], bins=10, ax=axes[1, 1])  # axes[1,1] 右下图
    plt.show()

    # 数据探索：相关性
    fig2, axes = plt.subplots(2, 2, figsize=(10, 8))  # 创建一个 2行 2列的画布
    sns.regplot(x=dfData['price'], y=dfData['sales'], ax=axes[0, 0])
    sns.regplot(x=dfData['average'], y=dfData['sales'], ax=axes[0, 1])
    sns.regplot(x=dfData['advertise'], y=dfData['sales'], ax=axes[1, 0])
    sns.regplot(x=dfData['difference'], y=dfData['sales'], ax=axes[1, 1])
    plt.show()

    # 数据探索：考察自变量平方项的相关性
    fig3, axes = plt.subplots(1, 2, figsize=(10, 4))  # 创建一个 1行 2列的画布
    sns.regplot(x=dfData['advertise'], y=dfData['sales'], order=2, ax=axes[0])  # order=2, 按 y=b*x**2 回归
    sns.regplot(x=dfData['difference'], y=dfData['sales'], order=2, ax=axes[1])
    plt.show()

    # 线性回归：分析因变量 Y(sales) 与 自变量 X1(Price diffrence)、X2(Advertise) 的关系
    y = dfData['sales']  # 根据因变量列名 list，建立 因变量数据集
    x0 = np.ones(dfData.shape[0])  # 截距列 x0=[1,...1]
    x1 = dfData['difference']  # 价格差，x4 = x1 - x2
    x2 = dfData['advertise']  # 广告费
    x3 = dfData['price']  # 销售价格
    x4 = dfData['average']  # 市场均价
    x5 = x2 ** 2  # 广告费的二次方
    x6 = x1 * x2  # 考察两个变量的相互作用

    # Model 1：Y = b0 + b1*X1 + b2*X2 + e
    # # 线性回归：分析因变量 Y(sales) 与 自变量 X1(Price diffrence)、X2(Advertise) 的关系
    X = np.column_stack((x0, x1, x2))  # [x0,x1,x2]
    Model1 = sm.OLS(y, X)  # 建立 OLS 模型: Y = b0 + b1*X1 + b2*X2 + e
    result1 = Model1.fit()  # 返回模型拟合结果
    yFit1 = result1.fittedvalues  # 模型拟合的 y 值
    prstd, ivLow, ivUp = wls_prediction_std(result1)  # 返回标准偏差和置信区间
    print(result1.summary())  # 输出回归分析的摘要
    print("\nModel1: Y = b0 + b1*X + b2*X2")
    print('Parameters: ', result1.params)  # 输出：拟合模型的系数

    # # Model 2：Y = b0 + b1*X1 + b2*X2 + b3*X3 + b4*X4 + e
    # 线性回归：分析因变量 Y(sales) 与 自变量 X1~X4 的关系
    X = np.column_stack((x0, x1, x2, x3, x4))  # [x0,x1,x2,...,x4]
    Model2 = sm.OLS(y, X)  # 建立 OLS 模型: Y = b0 + b1*X1 + b2*X2 + b3*X3 + e
    result2 = Model2.fit()  # 返回模型拟合结果
    yFit2 = result2.fittedvalues  # 模型拟合的 y 值
    prstd, ivLow, ivUp = wls_prediction_std(result2)  # 返回标准偏差和置信区间
    print(result2.summary())  # 输出回归分析的摘要
    print("\nModel2: Y = b0 + b1*X + ... + b4*X4")
    print('Parameters: ', result2.params)  # 输出：拟合模型的系数

    # # Model 3：Y = b0 + b1*X1 + b2*X2 + b3*X2**2 + e
    # # 线性回归：分析因变量 Y(sales) 与 自变量 X1、X2 及 X2平方（X5）的关系
    X = np.column_stack((x0, x1, x2, x5))  # [x0,x1,x2,x2**2]
    Model3 = sm.OLS(y, X)  # 建立 OLS 模型: Y = b0 + b1*X1 + b2*X2 + b3*X2**2 + e
    result3 = Model3.fit()  # 返回模型拟合结果
    yFit3 = result3.fittedvalues  # 模型拟合的 y 值
    prstd, ivLow, ivUp = wls_prediction_std(result3)  # 返回标准偏差和置信区间
    print(result3.summary())  # 输出回归分析的摘要
    print("\nModel3: Y = b0 + b1*X1 + b2*X2 + b3*X2**2")
    print('Parameters: ', result3.params)  # 输出：拟合模型的系数

    # 拟合结果绘图
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(range(len(y)), y, 'b-.', label='Sample')  # 样本数据
    ax.plot(range(len(y)), yFit3, 'r-', label='Fitting')  # 拟合数据
    # ax.plot(range(len(y)), yFit2, 'm--', label='fitting')  # 拟合数据
    ax.plot(range(len(y)), ivUp, '--', color='pink', label="ConfR")  # 95% 置信区间 上限
    ax.plot(range(len(y)), ivLow, '--', color='pink')  # 95% 置信区间 下限
    ax.legend(loc='best')  # 显示图例
    plt.title('Regression analysis with sales of toothpaste')
    plt.xlabel('period')
    plt.ylabel('sales')
    plt.show()
    return


if __name__ == '__main__':
    main()
