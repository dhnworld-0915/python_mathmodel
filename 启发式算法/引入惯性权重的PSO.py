# 载入所需的包
import numpy as np
import random
import copy


# 初始化粒子种群
def initial(pop, dim, ub, lb):
    X = np.zeros([pop, dim])
    for i in range(pop):
        X[i, :] = np.random.uniform(low=lb[0], high=ub[0], size=(1, dim))
    return X, lb, ub


'''定义适应度函数'''


def fun(X):
    O = 0
    for i in X:
        O += i ** 2
    return O


'''计算适应度函数'''


def CaculateFitness(X, fun):
    pop = X.shape[0]
    fitness = np.zeros([pop, 1])
    for i in range(pop):
        fitness[i] = fun(X[i, :])
    return fitness


'''适应度排序'''


def SortFitness(Fit):
    fitness = np.sort(Fit, axis=0)
    index = np.argsort(Fit, axis=0)
    return fitness, index


'''根据适应度对位置进行排序'''


def SortPosition(X, index):
    Xnew = np.zeros(X.shape)
    for i in range(X.shape[0]):
        Xnew[i, :] = X[index[i], :]
    return Xnew


'''引入惯性权重的速度和位置更新'''


def VPUpdate(X, pop, c1, c2, w1, w2, t, Velocity, GbestPositon, Max_iter):
    X_new = copy.copy(X)
    Velocity_new = copy.copy(Velocity)
    dim = X.shape[1]
    w = (w1 - w2) * (Max_iter - t) / Max_iter + w2
    for i in range(pop):
        for j in range(dim):
            X_new[i, j] = X[i, j] + Velocity[0, j]
            Velocity_new[0, j] = w * Velocity[0, j] + c1 * np.random.randn() * (
                    X[0, j] - X[i, j]) + c2 * np.random.randn() * (GbestPositon[0, j] - X[i, j])
    return X_new, Velocity_new


'''边界检查函数'''


def BorderCheck(X, ub, lb, pop, dim):
    for i in range(pop):
        for j in range(dim):
            if X[i, j] > ub[j]:
                X[i, j] = np.random.rand() * (ub[j] - lb[j]) + lb[j]
            elif X[i, j] < lb[j]:
                X[i, j] = np.random.rand() * (ub[j] - lb[j]) + lb[j]
    return X


'''速度检查函数'''


def VelocityCheck(Velocity, Velocity_max, dim):
    for j in range(dim):
        if Velocity[0, j] > Velocity_max:
            Velocity[0, j] = np.random.rand() * Velocity_max
    return Velocity


'''粒子群优化算法'''


def PSO(pop, dim, lb, ub, c1, c2, w1, w2, Velocity_max, Max_iter, fun):
    X, lb, ub = initial(pop, dim, ub, lb)  # 初始化种群
    fitness = CaculateFitness(X, fun)  # 计算适应度值
    fitness, sortIndex = SortFitness(fitness)  # 对适应度值排序
    X = SortPosition(X, sortIndex)  # 种群排序
    Velocity = np.zeros([1, dim])
    GbestScore = copy.copy(fitness[0])
    GbestPositon = np.zeros([1, dim])
    GbestPositon[0, :] = copy.copy(X[0, :])
    Curve = np.zeros([Max_iter, 1])
    for i in range(Max_iter):
        X, Velocity = VPUpdate(X, pop, c1, c2, w1, w2, i, Velocity, GbestPositon, Max_iter)
        X = BorderCheck(X, ub, lb, pop, dim)  # 边界检测
        Velocity = VelocityCheck(Velocity, Velocity_max, dim)  # 速度检查
        fitness = CaculateFitness(X, fun)  # 计算适应度值
        fitness, sortIndex = SortFitness(fitness)  # 对适应度值排序
        X = SortPosition(X, sortIndex)  # 种群排序
        if fitness[0] <= GbestScore:  # 更新全局最优
            GbestScore = copy.copy(fitness[0])
            GbestPositon[0, :] = copy.copy(X[0, :])
        Curve[i] = GbestScore
    return GbestScore, GbestPositon, Curve


# #设置参数
pop = 50  # 种群数量
Max_iter = 500  # 最大迭代次数
dim = 2  # 维度
lb = np.min(-10) * np.ones([dim, 1])  # 下边界
ub = np.max(10) * np.ones([dim, 1])  # 上边界
Velocity_max = 1200  # 粒子最大速度
c1 = 2  # 学习因子
c2 = 2  # 学习因子
w1 = 0.9
w2 = 0.4
# 适应度函数选择
fobj = fun

GbestScore, GbestPositon, Curve = PSO(pop, dim, lb, ub, c1, c2, w1, w2,
                                      Velocity_max, Max_iter, fun)
print('PSO最优适应度值：', GbestScore)
print('PSO最优位置：', GbestPositon)
