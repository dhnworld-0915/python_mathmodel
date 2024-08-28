from scipy.integrate import odeint, solve_bvp
import numpy as np
import matplotlib.pyplot as plt


# 1. 求解微分方程组边值问题，DEMO
# y'' + abs(y) = 0, y(0)=0.5, y(4)=-1.5

# 导数函数，计算导数 dY/dx
def dydx(x, y):
    dy0 = y[1]
    dy1 = -abs(y[0])
    return np.vstack((dy0, dy1))


# 计算 边界条件
def boundCond(ya, yb):
    fa = 0.5  # 边界条件 y(xa=0) = 0.5
    fb = -1.5  # 边界条件 y(xb=4) = -1.5
    return np.array([ya[0] - fa, yb[0] - fb])


xa, xb = 0, 4  # 边界点 (xa,xb)
# fa, fb = 0.5, -1.5  # 边界点的 y值
xini = np.linspace(xa, xb, 11)  # 确定 x 的初值
yini = np.zeros((2, xini.size))  # 确定 y 的初值
res = solve_bvp(dydx, boundCond, xini, yini)  # 求解 BVP

xSol = np.linspace(xa, xb, 100)  # 输出的网格节点
ySol = res.sol(xSol)[0]  # 网格节点处的 y 值

plt.plot(xSol, ySol, label='y')
# plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("scipy.integrate.solve_bvp")
plt.show()

from scipy.integrate import odeint, solve_bvp
import numpy as np
import matplotlib.pyplot as plt


# 2. 求解微分方程边值问题，水滴的横截面
# 导数函数，计算 h=[h0,h1] 点的导数 dh/dx
def dhdx(x, h):
    # 计算 dh0/dx, dh1/dx 的值
    dh0 = h[1]  # 计算 dh0/dx
    dh1 = (h[0] - 1) * (1 + h[1] * h[1]) ** 1.5  # 计算 dh1/dx
    return np.vstack((dh0, dh1))


# 计算 边界条件
def boundCond(ha, hb):
    # ha = 0  # 边界条件：h0(x=-1) = 0
    # hb = 0  # 边界条件：h0(x=1) = 0
    return np.array([ha[0], hb[0]])


xa, xb = -1, 1  # 边界点 (xa=0, xb=1)
xini = np.linspace(xa, xb, 11)  # 设置 x 的初值
hini = np.zeros((2, xini.size))  # 设置 h 的初值

res = solve_bvp(dhdx, boundCond, xini, hini)  # 求解 BVP
# scipy.integrate.solve_bvp(fun, bc, x, y,..)
#   fun(x, y, ..), 导数函数 f(y,x)，y在 x 处的导数。
#   bc(ya, yb, ..), 边界条件，y 在两点边界的函数。
#   x: shape (m)，初始网格的序列，起止于两点边界值 xa，xb。
#   y: shape (n,m)，网格节点处函数值的初值，第 i 列对应于 x[i]。

xSol = np.linspace(xa, xb, 100)  # 输出的网格节点
hSol = res.sol(xSol)[0]  # 网格节点处的 h 值
plt.plot(xSol, hSol, label='h(x)')
plt.xlabel("x")
plt.ylabel("h(x)")
plt.axis([-1, 1, 0, 1])
plt.title("Cross section of water drop by BVP")
plt.show()

from scipy.integrate import odeint, solve_bvp
import numpy as np
import matplotlib.pyplot as plt


# 3. 求解微分方程组边值问题，Mathieu 方程
# y0' = y1, y1' = -(lam-2*q*cos(2x))y0)
# y0(0)=1, y1(0)=0, y1(pi)=0

# 导数函数，计算导数 dY/dx
def dydx(x, y, p):  # p 是待定参数
    lam = p[0]
    q = 10
    dy0 = y[1]
    dy1 = -(lam - 2 * q * np.cos(2 * x)) * y[0]
    return np.vstack((dy0, dy1))


# 计算 边界条件
def boundCond(ya, yb, p):
    lam = p[0]
    return np.array([ya[0] - 1, ya[0], yb[0]])


xa, xb = 0, np.pi  # 边界点 (xa,xb)
xini = np.linspace(xa, xb, 11)  # 确定 x 的初值
xSol = np.linspace(xa, xb, 100)  # 输出的网格节点

for k in range(5):
    A = 0.75 * k
    y0ini = np.cos(8 * xini)  # 设置 y0 的初值
    y1ini = -A * np.sin(8 * xini)  # 设置 y1 的初值
    yini = np.vstack((y0ini, y1ini))  # 确定 y=[y0,y1] 的初值
    res = solve_bvp(dydx, boundCond, xini, yini, p=[10])  # 求解 BVP
    y0 = res.sol(xSol)[0]  # 网格节点处的 y 值
    y1 = res.sol(xSol)[1]  # 网格节点处的 y 值
    plt.plot(xSol, y0, '--')
    plt.plot(xSol, y1, '-', label='A = {:.2f}'.format(A))

plt.xlabel("x")
plt.ylabel("y")
plt.title("Characteristic function of Mathieu equation")
plt.axis([0, np.pi, -5, 5])
plt.legend(loc='best')
plt.text(2, -4, color='whitesmoke')
plt.show()
# 初值 A从 0~3.0 变化时，y-x 曲线（图中虚线）几乎不变，但 y’-x 的振幅增大；当 A 再稍微增大，系统就进入不稳定区， y-x 曲线振荡发散（图中未表示）。