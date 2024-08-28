# 模拟退火算法解TSP问题
import numpy as np
import random as rd


def lengthCal(path, distmat):  # 计算距离
    length = 0
    for i in range(len(path) - 1):
        length += distmat[path[i]][path[i + 1]]
    length += distmat[path[-1]][path[0]]
    return length


def exchange(path, exchangeSeq):  # 交换路径中的两点得到新路径
    newPath = []
    for i in range(len(path)):
        newPath.append(path[i])
    temp = newPath[exchangeSeq[0]]
    newPath[exchangeSeq[0]] = newPath[exchangeSeq[1]]
    newPath[exchangeSeq[1]] = temp
    return newPath


distmat = np.array([[0, 35, 29, 67, 60, 50, 66, 44, 72, 41, 48, 97],
                    [35, 0, 34, 36, 28, 37, 55, 49, 78, 76, 70, 110],
                    [29, 34, 0, 58, 41, 63, 79, 68, 103, 69, 78, 130],
                    [67, 36, 58, 0, 26, 38, 61, 80, 87, 110, 100, 110],
                    [60, 28, 41, 26, 0, 61, 78, 73, 103, 100, 96, 130],
                    [50, 37, 63, 38, 61, 0, 16, 64, 50, 95, 81, 95],
                    [66, 55, 79, 61, 78, 16, 0, 49, 34, 82, 68, 83],
                    [44, 49, 68, 80, 73, 64, 49, 0, 35, 43, 30, 62],
                    [72, 78, 103, 87, 103, 50, 34, 35, 0, 47, 32, 48],
                    [41, 76, 69, 110, 100, 95, 82, 43, 47, 0, 26, 74],
                    [48, 70, 78, 100, 96, 81, 68, 30, 32, 26, 0, 58],
                    [97, 110, 130, 110, 130, 95, 83, 62, 48, 74, 58, 0]])

T = 100  # 初始温度
α = 0.97  # 温度变化率
iters = 1000  # 每个温度的迭代次数
path = [i for i in range(12)]  # 随机初始化路径
rd.shuffle(path)  # shuffle()方法将序列的所有元素随机排序
while T > 10:
    for i in range(iters):
        exchangeSeq = rd.sample(range(0, 12), 2)  # 从序列中随机抽取2个元素，并将2个元素以list形式返回
        newPath = exchange(path, exchangeSeq)  # 随机交换路径中的两个点
        distanceDif = lengthCal(newPath, distmat) - lengthCal(path, distmat)
        if distanceDif < 0:
            path = newPath  # 接受新的解
        else:  # 以概率exp(-ΔT/T)接受新的解
            if rd.random() < np.exp(- distanceDif / T):  # random()方法返回随机生成的一个实数，它在[0,1)范围内
                path = newPath
    T = α * T
print("满意解为")
print(path)
print("距离为")
print(lengthCal(path, distmat))

import math
from random import random
import matplotlib.pyplot as plt


def func(x, y):  # 函数优化问题
    res = 4 * x ** 2 - 2.1 * x ** 4 + x ** 6 / 3 + x * y - 4 * y ** 2 + 4 * y ** 4
    return res


# x为公式里的x1,y为公式里面的x2
class SA:
    def __init__(self, func, iter=100, T0=100, Tf=0.01, alpha=0.99):
        self.func = func
        self.iter = iter  # 内循环迭代次数,即为L =100
        self.alpha = alpha  # 降温系数，alpha=0.99
        self.T0 = T0  # 初始温度T0为100
        self.Tf = Tf  # 温度终值Tf为0.01
        self.T = T0  # 当前温度
        self.x = [random() * 11 - 5 for i in range(iter)]  # 随机生成100个x的值
        self.y = [random() * 11 - 5 for i in range(iter)]  # 随机生成100个y的值
        self.most_best = []
        """
        random()这个函数取0到1之间的小数
        如果你要取0-10之间的整数（包括0和10）就写成 (int)random()*11就可以了，11乘以零点多的数最大是10点多，最小是0点多
        该实例中x1和x2的绝对值不超过5（包含整数5和-5），（random() * 11 -5）的结果是-6到6之间的任意值（不包括-6和6）
        （random() * 10 -5）的结果是-5到5之间的任意值（不包括-5和5），所有先乘以11，取-6到6之间的值，产生新解过程中，用一个if条件语句把-5到5之间（包括整数5和-5）的筛选出来。
        """
        self.history = {'f': [], 'T': []}

    def generate_new(self, x, y):  # 扰动产生新解的过程
        while True:
            x_new = x + self.T * (random() - random())
            y_new = y + self.T * (random() - random())
            if (-5 <= x_new <= 5) & (-5 <= y_new <= 5):
                break  # 重复得到新解，直到产生的新解满足约束条件
        return x_new, y_new

    def Metrospolis(self, f, f_new):  # Metropolis准则
        if f_new <= f:
            return 1
        else:
            p = math.exp((f - f_new) / self.T)
            if random() < p:
                return 1
            else:
                return 0

    def best(self):  # 获取最优目标函数值
        f_list = []  # f_list数组保存每次迭代之后的值
        for i in range(self.iter):
            f = self.func(self.x[i], self.y[i])
            f_list.append(f)
        f_best = min(f_list)

        idx = f_list.index(f_best)
        return f_best, idx  # f_best,idx分别为在该温度下，迭代L次之后目标函数的最优解和最优解的下标

    def run(self):
        count = 0
        # 外循环迭代，当前温度小于终止温度的阈值
        while self.T > self.Tf:

            # 内循环迭代100次
            for i in range(self.iter):
                f = self.func(self.x[i], self.y[i])  # f为迭代一次后的值
                x_new, y_new = self.generate_new(self.x[i], self.y[i])  # 产生新解
                f_new = self.func(x_new, y_new)  # 产生新值
                if self.Metrospolis(f, f_new):  # 判断是否接受新值
                    self.x[i] = x_new  # 如果接受新值，则把新值的x,y存入x数组和y数组
                    self.y[i] = y_new
            # 迭代L次记录在该温度下最优解
            ft, _ = self.best()
            self.history['f'].append(ft)
            self.history['T'].append(self.T)
            # 温度按照一定的比例下降（冷却）
            self.T = self.T * self.alpha
            count += 1

            # 得到最优解
        f_best, idx = self.best()
        print(f"F={f_best}, x={self.x[idx]}, y={self.y[idx]}")


sa = SA(func)
sa.run()

plt.plot(sa.history['T'], sa.history['f'])
plt.title('SA')
plt.xlabel('T')
plt.ylabel('f')
plt.gca().invert_xaxis()
plt.show()
