"""GM(1,1)模型"""

import numpy as np
import pandas as pd
from scipy import io, integrate, linalg, signal
from scipy.sparse.linalg import eigs
from scipy.integrate import odeint
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 步骤一（替换sans-serif字体）
plt.rcParams['axes.unicode_minus'] = False  # 步骤二（解决坐标轴负数的负号显示问题）


# 参考：https://www.cnblogs.com/jjmg/p/grey_model_by_python.html
# 其他案例：https://github.com/dontLoveBugs/GM-1-1

# 线性平移预处理，确保数据级比在可容覆盖范围
def greyModelPreprocess(dataVec):
    """Set linear-bias c for dataVec"""

    c = 0
    x0 = np.array(dataVec, float)
    n = x0.shape[0]  # 行数

    # 确定数值上下限
    L = np.exp(-2 / (n + 1))
    R = np.exp(2 / (n + 2))
    xmax = x0.max()
    xmin = x0.min()
    if xmin < 1:
        x0 += (1 - xmin)
        c += (1 - xmin)
    xmax = x0.max()
    xmin = x0.min()
    lambda_ = x0[0:-1] / x0[1:]  # 计算级比
    lambda_max = lambda_.max()
    lambda_min = lambda_.min()
    while lambda_max > R or lambda_min < L:
        x0 += xmin
        c += xmin
        xmax = x0.max()
        xmin = x0.min()
        lambda_ = x0[0:-1] / x0[1:]
        lambda_max = lambda_.max()
        lambda_min = lambda_.min()
    return c


# 灰色预测模型
def greyModel(dataVec, predictLen):
    """Grey Model for exponential prediction"""
    # dataVec = [1, 2, 3, 4, 5, 6]
    # predictLen = 5

    x0 = np.array(dataVec, float)
    n = x0.shape[0]
    x1 = np.cumsum(x0)
    B = np.array([-0.5 * (x1[0:-1] + x1[1:]), np.ones(n - 1)]).T
    Y = x0[1:]
    u = linalg.lstsq(B, Y)[0]

    def diffEqu(y, t, a, b):
        return np.array(-a * y + b)

    t = np.arange(n + predictLen)
    sol = odeint(diffEqu, x0[0], t, args=(u[0], u[1]))
    sol = sol.squeeze()
    res = np.hstack((x0[0], np.diff(sol)))
    return res


# 输入数据
x = np.array([-18, 0.34, 4.68, 8.49, 29.84, 50.21, 77.65, 109.36])
c = greyModelPreprocess(x)
x_hat = greyModel(x + c, 5) - c

# 画图
t1 = range(x.size)
t2 = range(x_hat.size)
plt.plot(t1, x, color='r', linestyle="-", marker='*', label='True')
plt.plot(t2, x_hat, color='b', linestyle="--", marker='.', label="Predict")
plt.legend(loc='upper right')
plt.xlabel('xlabel')
plt.ylabel('ylabel')
plt.title('Prediction by Grey Model (GM(1,1))')
plt.show()

"""GM(1,N)模型"""

# 参考：https://aitechtogether.com/article/16099.html

from decimal import *


class GM11:
    def __init__(self):
        self.f = None

    @staticmethod
    def isUsable(X0):
        """判断是否通过光滑检验"""
        X1 = X0.cumsum()
        rho = [X0[i] / X1[i - 1] for i in range(1, len(X0))]
        rho_ratio = [rho[i + 1] / rho[i] for i in range(len(rho) - 1)]
        print(rho, rho_ratio)
        flag = True
        for i in range(2, len(rho) - 1):
            if rho[i] > 0.5 or rho[i + 1] / rho[i] >= 1:
                flag = False
        if rho[-1] > 0.5:
            flag = False
        if flag:
            print("数据通过光滑校验")
        else:
            print("该数据未通过光滑校验")

        '''判断是否通过级比检验'''
        lambds = [X0[i - 1] / X0[i] for i in range(1, len(X0))]
        X_min = np.e ** (-2 / (len(X0) + 1))
        X_max = np.e ** (2 / (len(X0) + 1))
        for lambd in lambds:
            if lambd < X_min or lambd > X_max:
                print('该数据未通过级比检验')
                return
        print('该数据通过级比检验')

    def train(self, X0):
        X1 = X0.cumsum(axis=0)  # [x_2^1,x_3^1,...,x_n^1,x_1^1] # 其中x_i^1为x_i^01次累加后的列向量
        Z = (-0.5 * (X1[:, -1][:-1] + X1[:, -1][1:])).reshape(-1, 1)
        # 数据矩阵(matrix) A、B
        A = (X0[:, -1][1:]).reshape(-1, 1)
        B = np.hstack((Z, X1[1:, :-1]))
        print('Z: ', Z.shape, 'B', B.shape, 'X1', X1.shape)
        # 求参数
        u = np.linalg.inv(np.matmul(B.T, B)).dot(B.T).dot(A)
        a = u[0][0]
        b = u[1:]
        print("灰参数a：", a, "，参数矩阵(matrix)b：", b.shape)
        self.f = lambda k, X1: (X0[0, -1] - (1 / a) * (X1[k]).dot(b)) * np.exp(-a * k) + (1 / a) * (X1[k]).dot(b)

    def predict(self, k, X0):
        """
        :param k: k为预测的第k个值
        :param X0: X0为【k*n】的矩阵(matrix),n为特征的个数，k为样本的个数
        :return:
        """
        X1 = X0.cumsum(axis=0)
        X1_hat = [float(self.f(k, X1)[0]) for k in range(k)]
        X0_hat = np.diff(X1_hat)
        X0_hat = np.hstack((X1_hat[0], X0_hat))
        return X0_hat

    @staticmethod
    def evaluate(X0_hat, X0):
        """
        根据后验差比及小误差概率判断预测结果
        :param X0_hat: 预测结果
        :return:
        """
        S1 = np.std(X0, ddof=1)  # 原始数据样本标准差(standard deviation)
        S2 = np.std(X0 - X0_hat, ddof=1)  # 残差数据样本标准差(standard deviation)
        C = S2 / S1  # 后验差比
        Pe = np.mean(X0 - X0_hat)
        temp = np.abs((X0 - X0_hat - Pe)) < 0.6745 * S1
        p = np.count_nonzero(temp) / len(X0)  # 计算小误差概率
        print("原数据样本标准差(standard deviation)：", S1)
        print("残差样本标准差(standard deviation)：", S2)
        print("后验差：", C)
        print("小误差概率p：", p)


data = pd.read_csv("../data&result/water.csv")
# data.drop('供水总量', axis=1, inplace=True)
# 原始数据X
X = data.values
# 训练集
X_train = X[:, :]
# 测试集
X_test = []

model = GM11()
model.isUsable(X_train[:, -1])  # 判断模型可行性
model.train(X_train)  # 训练
Y_pred = model.predict(len(X), X[:, :-1])  # 预测
Y_train_pred = Y_pred[:len(X_train)]
Y_test_pred = Y_pred[len(X_train):]
print(model.evaluate(Y_train_pred, X_train[:, -1]))  # 评估)
# score_test = model.evaluate(Y_test_pred, X_test[:, -1])

# 可视化
plt.grid()
plt.plot(np.arange(len(Y_train_pred)), X_train[:, -1], '->')
plt.plot(np.arange(len(Y_train_pred)), Y_train_pred, '-o')
plt.legend(['负荷实际值', '灰色预测模型预测值'])
plt.title('训练集')
plt.show()

# # 可视化
# plt.grid()
# plt.plot(np.arange(len(Y_test_pred)), X_test[:, -1], '->')
# plt.plot(np.arange(len(Y_test_pred)), Y_test_pred, '-o')
# plt.legend(['负荷实际值', '灰色预测模型预测值'])
# plt.title('测试集')
# plt.show()
