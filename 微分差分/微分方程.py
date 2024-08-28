# 1. 求解微分方程初值问题(scipy.integrate.odeint)

from scipy.integrate import odeint  # 导入 scipy.integrate 模块
import numpy as np
import matplotlib.pyplot as plt


def dy_dt(y, t):  # 定义函数 f(y,t)
    return np.sin(t ** 2)


y0 = [1]  # y0 = 1 也可以
t = np.arange(-10, 10, 0.01)  # (start,stop,step)
y = odeint(dy_dt, y0, t)  # 求解微分方程初值问题

# 绘图
plt.plot(t, y)
plt.title("scipy.integrate.odeint")
plt.show()

# 2. 求解微分方程组初值问题(scipy.integrate.odeint)
from scipy.integrate import odeint  # 导入 scipy.integrate 模块
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# 导数函数, 求 W=[x,y,z] 点的导数 dW/dt
def lorenz(W, t, p, r, b):
    x, y, z = W  # W=[x,y,z]
    dx_dt = p * (y - x)  # dx/dt = p*(y-x), p: sigma
    dy_dt = x * (r - z) - y  # dy/dt = x*(r-z)-y, r:rho
    dz_dt = x * y - b * z  # dz/dt = x*y - b*z, b;beta
    return np.array([dx_dt, dy_dt, dz_dt])


t = np.arange(0, 30, 0.01)  # 创建时间点 (start,stop,step)
paras = (10.0, 28.0, 3.0)  # 设置 Lorenz 方程中的参数 (p,r,b)

# 调用ode对lorenz进行求解, 用两个不同的初始值 W1、W2 分别求解
W1 = (0.0, 1.00, 0.0)  # 定义初值为 W1
track1 = odeint(lorenz, W1, t, args=(10.0, 28.0, 3.0))  # args 设置导数函数的参数
W2 = (0.0, 1.01, 0.0)  # 定义初值为 W2
track2 = odeint(lorenz, W2, t, args=paras)  # 通过 paras 传递导数函数的参数

# 绘图
fig = plt.figure()
ax = Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax)
ax.plot(track1[:, 0], track1[:, 1], track1[:, 2], color='magenta')  # 绘制轨迹 1
ax.plot(track2[:, 0], track2[:, 1], track2[:, 2], color='deepskyblue')  # 绘制轨迹 2
ax.set_title("Lorenz attractor by scipy.integrate.odeint")
plt.show()

# 3. 求解二阶微分方程初值问题(scipy.integrate.odeint)
# Second ODE by scipy.integrate.odeint
from scipy.integrate import odeint  # 导入 scipy.integrate 模块
import numpy as np
import matplotlib.pyplot as plt


# 导数函数，求 Y=[u,v] 点的导数 dY/dt
def deriv(Y, t, a, w):
    u, v = Y  # Y=[u,v]
    dY_dt = [v, -2 * a * v - w * w * u]
    return dY_dt


t = np.arange(0, 20, 0.01)  # 创建时间点 (start,stop,step)
# 设置导数函数中的参数 (a, w)
paras1 = (1, 0.6)  # 过阻尼：a^2 - w^2 > 0
paras2 = (1, 1)  # 临界阻尼：a^2 - w^2 = 0
paras3 = (0.3, 1)  # 欠阻尼：a^2 - w^2 < 0

# 调用ode对进行求解, 用两个不同的初始值 W1、W2 分别求解
Y0 = (1.0, 0.0)  # 定义初值为 Y0=[u0,v0]
Y1 = odeint(deriv, Y0, t, args=paras1)  # args 设置导数函数的参数
Y2 = odeint(deriv, Y0, t, args=paras2)  # args 设置导数函数的参数
Y3 = odeint(deriv, Y0, t, args=paras3)  # args 设置导数函数的参数
# W2 = (0.0, 1.01, 0.0)  # 定义初值为 W2
# track2 = odeint(lorenz, W2, t, args=paras)  # 通过 paras 传递导数函数的参数

# 绘图
plt.plot(t, Y1[:, 0], 'r-', label='u1(t)')
plt.plot(t, Y2[:, 0], 'b-', label='u2(t)')
plt.plot(t, Y3[:, 0], 'g-', label='u3(t)')
plt.plot(t, Y1[:, 1], 'r:', label='v1(t)')
plt.plot(t, Y2[:, 1], 'b:', label='v2(t)')
plt.plot(t, Y3[:, 1], 'g:', label='v3(t)')
plt.axis([0, 20, -0.8, 1.2])
plt.legend(loc='best')
plt.title("Second ODE by scipy.integrate.odeint")
plt.show()
# 结果讨论：
#
# RLC串联电路是典型的二阶系统，在零输入条件下根据 α 与 ω 的关系，电路的输出响应存在四种情况：
# 过阻尼： α 2 − ω 2 > 0 ,有 2 个不相等的负实数根；
# 临界阻尼： α 2 − ω 2 = 0 ,有 2 个相等的负实数根；
# 欠阻尼： α 2 − ω 2 < 0 ,有一对共轭复数根；
# 无阻尼：R = 0 ,有一对纯虚根。
# 例程中所选择的 3 组参数分别对应过阻尼、临界阻尼和欠阻尼的条件，微分方程的数值结果很好地体现了不同情况的相应曲线。
