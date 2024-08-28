from scipy.optimize import brent, fmin_ncg, minimize
import numpy as np


# 1. Demo1：单变量无约束优化问题(Scipy.optimize.brent)
# brent() 函数是 SciPy.optimize 模块中求解单变量无约束优化问题最小值的首选方法。这是牛顿法和二分法的混合方法，既能保证稳定性又能快速收敛。

def objf(x):  # 目标函数
    fx = x ** 2 - 8 * np.sin(2 * x + np.pi)
    return fx


xIni = -5.0
xOpt = brent(objf, brack=(xIni, 2))
print("xIni={:.4f}\tfxIni={:.4f}".format(xIni, objf(xIni)))
print("xOpt={:.4f}\tfxOpt={:.4f}".format(xOpt, objf(xOpt)))

from scipy.optimize import brent, fmin, minimize
import numpy as np


# 2. Demo2：多变量无约束优化问题(Scipy.optimize.brent)
# Rosenbrock 测试函数
def objf2(x):  # Rosenbrock benchmark function
    fx = sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)
    return fx


xIni = np.array([-2, -2])
xOpt = fmin(objf2, xIni)
print("xIni={:.4f},{:.4f}\tfxIni={:.4f}".format(xIni[0], xIni[1], objf2(xIni)))
print("xOpt={:.4f},{:.4f}\tfxOpt={:.4f}".format(xOpt[0], xOpt[1], objf2(xOpt)))

from scipy.optimize import brent, fmin, minimize
import numpy as np


# 3. Demo3：多变量边界约束优化问题(Scipy.optimize.minimize)
# 定义目标函数
def objf3(x):  # Rosenbrock 测试函数
    fx = sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)
    return fx


# 定义边界约束（优化变量的上下限）
b0 = (0.0, None)  # 0.0 <= x[0] <= Inf
b1 = (0.0, 10.0)  # 0.0 <= x[1] <= 10.0
b2 = (-5.0, 100.)  # -5.0 <= x[2] <= 100.0
bnds = (b0, b1, b2)  # 边界约束

# 优化计算
xIni = np.array([1., 2., 3.])
resRosen = minimize(objf3, xIni, method='SLSQP', bounds=bnds)
xOpt = resRosen.x

print("xOpt = {:.4f}, {:.4f}, {:.4f}".format(xOpt[0], xOpt[1], xOpt[2]))
print("min f(x) = {:.4f}".format(objf3(xOpt)))

from scipy.optimize import brent, fmin, minimize
import numpy as np


# 4. Demo4：约束非线性规划问题(Scipy.optimize.minimize)

# 程序说明：
# 在本例程中，目标函数中的参数 a, b, c, d 在子程序中直接赋值，这种实现方式最简单；
# 定义边界约束，即优化变量的上下限，与 3.2 中的例程相同，用 minimize() 函数中的选项 bounds=bnds 进行定义。
# 定义约束条件：
# 本案例有 4个约束条件，2个等式约束、2个不等式约束，上节中已写成标准形式；
# 本例程将每个约束条件作为一个子函数定义，
# minimize() 函数对约束条件按照字典格式： {‘type’: ‘ineq’, ‘fun’: functionname} 进行定义。‘type’ 的键值可选 ‘eq’ 和 ‘ineq’，分别表示的是约束和不等式约束；functionname是定义约束条件的函数名。
# 求解最小化问题 res，其中目标函数 objF4 和搜索的初值点 x0 是必需的，指定优化方法和边界条件、约束条件是可选项。
# 通过调用最小化问题的返回值可以得到优化是否成功的说明（res.message）、自变量的优化值（res.x）和目标函数的优化值（res.fun）。

def objF4(x):  # 定义目标函数
    a, b, c, d = 1, 2, 3, 8
    fx = a * x[0] ** 2 + b * x[1] ** 2 + c * x[2] ** 2 + d
    return fx


# 定义约束条件函数
def constraint1(x):  # 不等式约束 f(x)>=0
    return x[0] ** 2 - x[1] + x[2] ** 2


def constraint2(x):  # 不等式约束 转换为标准形式
    return -(x[0] + x[1] ** 2 + x[2] ** 3 - 20)


def constraint3(x):  # 等式约束
    return -x[0] - x[1] ** 2 + 2


def constraint4(x):  # 等式约束
    return x[1] + 2 * x[2] ** 2 - 3


# 定义边界约束
b = (0.0, None)
bnds = (b, b, b)

# 定义约束条件
con1 = {'type': 'ineq', 'fun': constraint1}
con2 = {'type': 'ineq', 'fun': constraint2}
con3 = {'type': 'eq', 'fun': constraint3}
con4 = {'type': 'eq', 'fun': constraint4}
cons = ([con1, con2, con3, con4])  # 4个约束条件

# 求解优化问题
x0 = np.array([1., 2., 3.])  # 定义搜索的初值
res = minimize(objF4, x0, method='SLSQP', bounds=bnds, constraints=cons)

print("Optimization problem (res):\t{}".format(res.message))  # 优化是否成功
print("xOpt = {}".format(res.x))  # 自变量的优化值
print("min f(x) = {:.4f}".format(res.fun))  # 目标函数的优化值

from scipy.optimize import brent, fmin, minimize
import numpy as np


# 5. Demo5：约束非线性规划问题(Scipy.optimize.minimize)

# 程序说明：
# 本例程的问题与 4.2 中的例程 1 是相同的，结果也相同，但编程实现的方法进行了改进；
# 本例程中目标函数中的参数 a, b, c, d 在主程序中赋值，通过 args 把参数传递到子程序，这种实现方式使参数赋值更为灵活，特别是适用于可变参数的问题；注意目标函数的定义不是 def objF5(x,args)，而是 def objF5(args)，要特别注意目标函数的定义和实现方法。
# 定义约束条件：
# 本案例有 4 个约束条件，2个等式约束、2个不等式约束，上节中已写成标准形式；
# 本例程将 4 个约束条件放在一个子函数中定义，是程序更加简洁。
# 注意每个约束条件仍然按照字典格式 {‘type’: ‘ineq’, ‘fun’: functionname} 进行定义，但 functionname 并不是函数名，而是一个 lambda 匿名函数。
# 通过调用最小化问题的返回值可以得到优化是否成功的说明（res.message）、自变量的优化值（res.x）和目标函数的优化值（res.fun）。

def objF5(args):  # 定义目标函数
    a, b, c, d = args
    fx = lambda x: a * x[0] ** 2 + b * x[1] ** 2 + c * x[2] ** 2 + d
    return fx


def constraint1():  # 定义约束条件函数
    cons = ({'type': 'ineq', 'fun': lambda x: (x[0] ** 2 - x[1] + x[2] ** 2)},  # 不等式约束 f(x)>=0
            {'type': 'ineq', 'fun': lambda x: -(x[0] + x[1] ** 2 + x[2] ** 3 - 20)},  # 不等式约束 转换为标准形式
            {'type': 'eq', 'fun': lambda x: (-x[0] - x[1] ** 2 + 2)},  # 等式约束
            {'type': 'eq', 'fun': lambda x: (x[1] + 2 * x[2] ** 2 - 3)})  # 等式约束
    return cons


# 定义边界约束
b = (0.0, None)
bnds = (b, b, b)
# 定义约束条件
cons = constraint1()
args1 = (1, 2, 3, 8)  # 定义目标函数中的参数
# 求解优化问题
x0 = np.array([1., 2., 3.])  # 定义搜索的初值
res1 = minimize(objF5(args1), x0, method='SLSQP', bounds=bnds, constraints=cons)

print("Optimization problem (res1):\t{}".format(res1.message))  # 优化是否成功
print("xOpt = {}".format(res1.x))  # 自变量的优化值
print("min f(x) = {:.4f}".format(res1.fun))  # 目标函数的优化值

from scipy.optimize import brent, fmin, minimize
import numpy as np


# 6. Demo6：约束非线性规划问题(Scipy.optimize.minimize)

# 程序说明：
# 本例程的问题与 4.3 中的例程 2 是相同的，结果也相同，但编程实现的方法进行了改进；
# 本例程中约束条件中的参数在主程序中赋值，通过 args 把参数传递到约束条件定义的子程序，这种实现方式使参数赋值更为灵活，特别是适用于可变参数的问题。
# 本例程中将边界约束条件即自变量的取值范围作为不等式约束条件处理，不另作边界条件设置。
# 通过调用最小化问题的返回值可以得到优化是否成功的说明（res.message）、自变量的优化值（res.x）和目标函数的优化值（res.fun）。

def objF6(args):  # 定义目标函数
    a, b, c, d = args
    fx = lambda x: a * x[0] ** 2 + b * x[1] ** 2 + c * x[2] ** 2 + d
    return fx


def constraint2(args):
    xmin0, xmin1, xmin2 = args
    cons = ({'type': 'ineq', 'fun': lambda x: (x[0] ** 2 - x[1] + x[2] ** 2)},  # 不等式约束 f(x)>=0
            {'type': 'ineq', 'fun': lambda x: -(x[0] + x[1] ** 2 + x[2] ** 3 - 20)},  # 不等式约束 转换为标准形式
            {'type': 'eq', 'fun': lambda x: (-x[0] - x[1] ** 2 + 2)},  # 等式约束
            {'type': 'eq', 'fun': lambda x: (x[1] + 2 * x[2] ** 2 - 3)},  # 等式约束
            {'type': 'ineq', 'fun': lambda x: (x[0] - xmin0)},  # x0 >= xmin0
            {'type': 'ineq', 'fun': lambda x: (x[1] - xmin1)},  # x1 >= xmin1
            {'type': 'ineq', 'fun': lambda x: (x[2] - xmin2)})  # x2 >= xmin2
    return cons


# 求解优化问题
args1 = (1, 2, 3, 8)  # 定义目标函数中的参数
args2 = (0.0, 0.0, 0.0)  # xmin0, xmin1, xmin2
cons2 = constraint2(args2)

x0 = np.array([1., 2., 3.])  # 定义搜索的初值
res2 = minimize(objF6(args1), x0, method='SLSQP', constraints=cons2)

print("Optimization problem (res2):\t{}".format(res2.message))  # 优化是否成功
print("xOpt = {}".format(res2.x))  # 自变量的优化值
print("min f(x) = {:.4f}".format(res2.fun))  # 目标函数的优化值
