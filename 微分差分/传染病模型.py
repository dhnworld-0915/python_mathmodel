# 传染病的数学模型是数学建模中的典型问题，标准名称是流行病的数学模型（Mathematical models of epidemic diseases）。建立传染病的数学模型来描述传染病的传播过程，研究传染病的传播速度、空间范围、传播途径、动力学机理等问题，以指导对传染病的有效地预防和控制，具有重要的现实意义。
# 不同类型传染病的传播具有不同的特点，传染病的传播模型不是从医学角度分析传染病的传播过程，而是按照传播机理建立不同的数学模型。
# 首先，把传染病流行范围内的人群分为 S、E、I、R 四类，具体含义如下：
# S 类（Susceptible），易感者，指缺乏免疫能力的健康人，与感染者接触后容易受到感染；
# E 类（Exposed），暴露者，指接触过感染者但暂无传染性的人，适用于存在潜伏期的传染病；
# I 类（Infectious），患病者，指具有传染性的患病者，可以传播给 S 类成员将其变为 E 类或 I 类成员；
# R 类（Recovered），康复者，指病愈后具有免疫力的人。如果免疫期有限，仍可以重新变为 S 类成员，进而被感染；如果是终身免疫，则不能再变为 S类、E类或 I 类成员。
# 常见的传染病模型按照传染病类型分为 SI、SIR、SIRS、SEIR 模型等，就是由以上四类人群根据不同传染病的特征进行组合而产生的不同模型。

# 1. SI 模型，常微分非常，解析解与数值解的比较
from scipy.integrate import odeint  # 导入 scipy.integrate 模块
import numpy as np  # 导入 numpy包
import matplotlib.pyplot as plt  # 导入 matplotlib包


def dy_dt(y, t, lamda, mu):  # 定义导数函数 f(y,t)
    dy_dt = lamda * y * (1 - y)  # di/dt = lamda*i*(1-i)
    return dy_dt


# 设置模型参数
number = 1e7  # 总人数
lamda = 1.0  # 日接触率, 患病者每天有效接触的易感者的平均人数
mu1 = 0.5  # 日治愈率, 每天被治愈的患病者人数占患病者总数的比例
y0 = i0 = 1e-6  # 患病者比例的初值
tEnd = 50  # 预测日期长度
t = np.arange(0.0, tEnd, 1)  # (start,stop,step)

yAnaly = 1 / (1 + (1 / i0 - 1) * np.exp(-lamda * t))  # 微分方程的解析解
yInteg = odeint(dy_dt, y0, t, args=(lamda, mu1))  # 求解微分方程初值问题
yDeriv = lamda * yInteg * (1 - yInteg)  # 微分方程的数值解

# 绘图
plt.plot(t, yAnaly, '-ob', label='analytic')
plt.plot(t, yInteg, ':.r', label='numerical')
plt.plot(t, yDeriv, '-g', label='dy_dt')
plt.title("Comparison between analytic and numerical solutions")
plt.legend(loc='right')
plt.axis((0, 50, -0.1, 1.1))
plt.show()

# 2. SIS 模型，常微分方程，解析解与数值解的比较
from scipy.integrate import odeint  # 导入 scipy.integrate 模块
import numpy as np  # 导入 numpy包
import matplotlib.pyplot as plt  # 导入 matplotlib包


def dy_dt(y, t, lamda, mu):  # SIS 模型，导数函数
    dy_dt = lamda * y * (1 - y) - mu * y  # di/dt = lamda*i*(1-i)-mu*i
    return dy_dt


# 设置模型参数
number = 1e5  # 总人数
lamda = 1.2  # 日接触率, 患病者每天有效接触的易感者的平均人数
sigma = 2.5  # 传染期接触数
mu = lamda / sigma  # 日治愈率, 每天被治愈的患病者人数占患病者总数的比例
fsig = 1 - 1 / sigma
y0 = i0 = 1e-5  # 患病者比例的初值
tEnd = 50  # 预测日期长度
t = np.arange(0.0, tEnd, 1)  # (start,stop,step)
print("lamda={}\tmu={}\tsigma={}\t(1-1/sig)={}".format(lamda, mu, sigma, fsig))

# 解析解
if lamda == mu:
    yAnaly = 1.0 / (lamda * t + 1.0 / i0)
else:
    yAnaly = 1.0 / ((lamda / (lamda - mu)) + ((1 / i0) - (lamda / (lamda - mu))) * np.exp(-(lamda - mu) * t))
# odeint 数值解，求解微分方程初值问题
ySI = odeint(dy_dt, y0, t, args=(lamda, 0))  # SI 模型
ySIS = odeint(dy_dt, y0, t, args=(lamda, mu))  # SIS 模型

# 绘图
plt.plot(t, yAnaly, '-ob', label='analytic')
plt.plot(t, ySIS, ':.r', label='ySIS')
plt.plot(t, ySI, '-g', label='ySI')

plt.title("Comparison between analytic and numerical solutions")
plt.axhline(y=fsig, ls="--", c='c')  # 添加水平直线
plt.legend(loc='best')
plt.axis((0, 50, -0.1, 1.1))
plt.show()

# 3. SIS 模型，模型参数对 di/dt的影响
from scipy.integrate import odeint  # 导入 scipy.integrate 模块
import numpy as np  # 导入 numpy包
import matplotlib.pyplot as plt  # 导入 matplotlib包


def dy_dt(y, t, lamda, mu):  # SIS 模型，导数函数
    dy_dt = lamda * y * (1 - y) - mu * y  # di/dt = lamda*i*(1-i)-mu*i
    return dy_dt


# 设置模型参数
number = 1e5  # 总人数
lamda = 1.2  # 日接触率, 患病者每天有效接触的易感者的平均人数
# sigma = np.array((0.1, 0.5, 0.8, 0.95, 1.0))  # 传染期接触数
sigma = np.array((0.5, 0.8, 1.0, 1.5, 2.0, 3.0))  # 传染期接触数
y0 = i0 = 0.05  # 患病者比例的初值
tEnd = 100  # 预测日期长度
t = np.arange(0.0, tEnd, 0.1)  # (start,stop,step)

for p in sigma:
    ySIS = odeint(dy_dt, y0, t, args=(lamda, lamda / p))  # SIS 模型
    yDeriv = lamda * ySIS * (1 - ySIS) - ySIS * lamda / p
    # plt.plot(t, yDeriv, '-', label=r"$\sigma$ = {}".format(p))
    plt.plot(ySIS, yDeriv, '-', label=r"$\sigma$ = {}".format(p))  # label='di/dt~i'
    print("lamda={}\tmu={}\tsigma={}\t(1-1/sig)={}".format(lamda, lamda / p, p, (1 - 1 / p)))

# 绘图
plt.axhline(y=0, ls="--", c='c')  # 添加水平直线
plt.title("i(t)~di/dt in SIS model")
plt.legend(loc='best')
plt.show()

# 4. SIR 模型，常微分方程组
from scipy.integrate import odeint  # 导入 scipy.integrate 模块
import numpy as np  # 导入 numpy包
import matplotlib.pyplot as plt  # 导入 matplotlib包


def dySIS(y, t, lamda, mu):  # SI/SIS 模型，导数函数
    dy_dt = lamda * y * (1 - y) - mu * y  # di/dt = lamda*i*(1-i)-mu*i
    return dy_dt


def dySIR(y, t, lamda, mu):  # SIR 模型，导数函数
    i, s = y
    di_dt = lamda * s * i - mu * i  # di/dt = lamda*s*i-mu*i
    ds_dt = -lamda * s * i  # ds/dt = -lamda*s*i
    return np.array([di_dt, ds_dt])


# 设置模型参数
number = 1e5  # 总人数
lamda = 0.2  # 日接触率, 患病者每天有效接触的易感者的平均人数
sigma = 2.5  # 传染期接触数
mu = lamda / sigma  # 日治愈率, 每天被治愈的患病者人数占患病者总数的比例
fsig = 1 - 1 / sigma
tEnd = 200  # 预测日期长度
t = np.arange(0.0, tEnd, 1)  # (start,stop,step)
i0 = 1e-4  # 患病者比例的初值
s0 = 1 - i0  # 易感者比例的初值
Y0 = (i0, s0)  # 微分方程组的初值

print("lamda={}\tmu={}\tsigma={}\t(1-1/sig)={}".format(lamda, mu, sigma, fsig))

# odeint 数值解，求解微分方程初值问题
ySI = odeint(dySIS, i0, t, args=(lamda, 0))  # SI 模型
ySIS = odeint(dySIS, i0, t, args=(lamda, mu))  # SIS 模型
ySIR = odeint(dySIR, Y0, t, args=(lamda, mu))  # SIR 模型

# 绘图
plt.title("Comparison among SI, SIS and SIR models")
plt.xlabel('t')
plt.axis((0, tEnd, -0.1, 1.1))
plt.axhline(y=0, ls="--", c='c')  # 添加水平直线
plt.plot(t, ySI, ':g', label='i(t)-SI')
plt.plot(t, ySIS, '--g', label='i(t)-SIS')
plt.plot(t, ySIR[:, 0], '-r', label='i(t)-SIR')
plt.plot(t, ySIR[:, 1], '-b', label='s(t)-SIR')
plt.plot(t, 1 - ySIR[:, 0] - ySIR[:, 1], '-m', label='r(t)-SIR')
plt.legend(loc='best')
plt.show()

# 5. SIR 模型，常微分方程组 相空间分析
from scipy.integrate import odeint  # 导入 scipy.integrate 模块
import numpy as np  # 导入 numpy包
import matplotlib.pyplot as plt  # 导入 matplotlib包


def dySIR(y, t, lamda, mu):  # SIR 模型，导数函数
    i, s = y
    di_dt = lamda * s * i - mu * i  # di/dt = lamda*s*i-mu*i
    ds_dt = -lamda * s * i  # ds/dt = -lamda*s*i
    return np.array([di_dt, ds_dt])


# 设置模型参数
number = 1e5  # 总人数
lamda = 0.2  # 日接触率, 患病者每天有效接触的易感者的平均人数
sigma = 2.5  # 传染期接触数
mu = lamda / sigma  # 日治愈率, 每天被治愈的患病者人数占患病者总数的比例
fsig = 1 - 1 / sigma
print("lamda={}\tmu={}\tsigma={}\t(1-1/sig)={}".format(lamda, mu, sigma, fsig))

# odeint 数值解，求解微分方程初值问题
tEnd = 200  # 预测日期长度
t = np.arange(0.0, tEnd, 1)  # (start,stop,step)
s0List = np.arange(0.01, 0.91, 0.1)  # (start,stop,step)
for s0 in s0List:  # s0, 易感者比例的初值
    i0 = 1 - s0  # i0, 患病者比例的初值
    Y0 = (i0, s0)  # 微分方程组的初值
    ySIR = odeint(dySIR, Y0, t, args=(lamda, mu))  # SIR 模型
    plt.plot(ySIR[:, 1], ySIR[:, 0])

# 绘图
plt.title("Phase trajectory of SIR models")
plt.axis([0, 1, 0, 1])
plt.plot([0, 1], [1, 0], 'b-')
plt.plot([1 / sigma, 1 / sigma], [0, 1 - 1 / sigma], 'b--')
plt.xlabel('s(t)')
plt.ylabel('i(t)')
plt.text(0.8, 0.9, r"$1/\sigma$ = {}".format(1 / sigma), color='b')
plt.show()

# 6. SEIR 模型，常微分方程组
from scipy.integrate import odeint  # 导入 scipy.integrate 模块
import numpy as np  # 导入 numpy包
import matplotlib.pyplot as plt  # 导入 matplotlib包


def dySIS(y, t, lamda, mu):  # SI/SIS 模型，导数函数
    dy_dt = lamda * y * (1 - y) - mu * y  # di/dt = lamda*i*(1-i)-mu*i
    return dy_dt


def dySIR(y, t, lamda, mu):  # SIR 模型，导数函数
    s, i = y  # youcans
    ds_dt = -lamda * s * i  # ds/dt = -lamda*s*i
    di_dt = lamda * s * i - mu * i  # di/dt = lamda*s*i-mu*i
    return np.array([ds_dt, di_dt])


def dySEIR(y, t, lamda, delta, mu):  # SEIR 模型，导数函数
    s, e, i = y
    ds_dt = -lamda * s * i  # ds/dt = -lamda*s*i
    de_dt = lamda * s * i - delta * e  # de/dt = lamda*s*i - delta*e
    di_dt = delta * e - mu * i  # di/dt = delta*e - mu*i
    return np.array([ds_dt, de_dt, di_dt])


# 设置模型参数
number = 1e5  # 总人数
lamda = 0.3  # 日接触率, 患病者每天有效接触的易感者的平均人数
delta = 0.03  # 日发病率，每天发病成为患病者的潜伏者占潜伏者总数的比例
mu = 0.06  # 日治愈率, 每天治愈的患病者人数占患病者总数的比例
sigma = lamda / mu  # 传染期接触数
fsig = 1 - 1 / sigma
tEnd = 300  # 预测日期长度
t = np.arange(0.0, tEnd, 1)  # (start,stop,step)
i0 = 1e-3  # 患病者比例的初值
e0 = 1e-3  # 潜伏者比例的初值
s0 = 1 - i0  # 易感者比例的初值
Y0 = (s0, e0, i0)  # 微分方程组的初值

# odeint 数值解，求解微分方程初值问题
ySI = odeint(dySIS, i0, t, args=(lamda, 0))  # SI 模型
ySIS = odeint(dySIS, i0, t, args=(lamda, mu))  # SIS 模型
ySIR = odeint(dySIR, (s0, i0), t, args=(lamda, mu))  # SIR 模型
ySEIR = odeint(dySEIR, Y0, t, args=(lamda, delta, mu))  # SEIR 模型

# 输出绘图
print("lamda={}\tmu={}\tsigma={}\t(1-1/sig)={}".format(lamda, mu, sigma, fsig))
plt.title("Comparison among SI, SIS, SIR and SEIR models")
plt.xlabel('t')
plt.axis((0, tEnd, -0.1, 1.1))
plt.plot(t, ySI, 'cadetblue', label='i(t)-SI')
plt.plot(t, ySIS, 'steelblue', label='i(t)-SIS')
plt.plot(t, ySIR[:, 1], 'cornflowerblue', label='i(t)-SIR')
# plt.plot(t, 1-ySIR[:,0]-ySIR[:,1], 'cornflowerblue', label='r(t)-SIR')
plt.plot(t, ySEIR[:, 0], '--', color='darkviolet', label='s(t)-SEIR')
plt.plot(t, ySEIR[:, 1], '-.', color='orchid', label='e(t)-SEIR')
plt.plot(t, ySEIR[:, 2], '-', color='m', label='i(t)-SEIR')
plt.plot(t, 1 - ySEIR[:, 0] - ySEIR[:, 1] - ySEIR[:, 2], ':', color='palevioletred', label='r(t)-SEIR')
plt.legend(loc='right')
plt.show()

# 7. SEIR 模型，常微分方程组 相空间分析: e(t)~i(t)
from scipy.integrate import odeint  # 导入 scipy.integrate 模块
import numpy as np  # 导入 numpy包
import matplotlib.pyplot as plt  # 导入 matplotlib包


def dySEIR(y, t, lamda, delta, mu):  # SEIR 模型，导数函数
    s, e, i = y
    ds_dt = -lamda * s * i  # ds/dt = -lamda*s*i
    de_dt = lamda * s * i - delta * e  # de/dt = lamda*s*i - delta*e
    di_dt = delta * e - mu * i  # di/dt = delta*e - mu*i
    return np.array([ds_dt, de_dt, di_dt])


# 设置模型参数
number = 1e5  # 总人数
lamda = 0.3  # 日接触率, 患病者每天有效接触的易感者的平均人数
delta = 0.1  # 日发病率，每天发病成为患病者的潜伏者占潜伏者总数的比例
mu = 0.1  # 日治愈率, 每天治愈的患病者人数占患病者总数的比例
sigma = lamda / mu  # 传染期接触数
tEnd = 500  # 预测日期长度
t = np.arange(0.0, tEnd, 1)  # (start,stop,step)# e0List = np.arange(0.01,0.4,0.05)  # (start,stop,step)

e0List = np.arange(0.01, 0.4, 0.05)  # (start,stop,step)
for e0 in e0List:
    # odeint 数值解，求解微分方程初值问题
    i0 = 0  # 潜伏者比例的初值
    s0 = 1 - i0 - e0  # 易感者比例的初值
    ySEIR = odeint(dySEIR, (s0, e0, i0), t, args=(lamda, delta, mu))  # SEIR 模型
    plt.plot(ySEIR[:, 1], ySEIR[:, 2])  # (e(t),i(t))
    print("lamda={}\tdelta={}\mu={}\tsigma={}\ti0={}\te0={}".format(lamda, delta, mu, lamda / mu, i0, e0))

# 输出绘图
plt.title("Phase trajectory of SEIR models: e(t)~i(t)")
plt.axis((0, 0.4, 0, 0.4))
plt.plot([0, 0.4], [0, 0.35], 'y--')  # [x1,x2][y1,y2]
plt.plot([0, 0.4], [0, 0.18], 'y--')  # [x1,x2][y1,y2]
plt.text(0.02, 0.36, r"$\lambda=0.3, \delta=0.1, \mu=0.1$", color='black')
plt.xlabel('e(t)')
plt.ylabel('i(t)')
plt.show()

# 8. SEIR2 模型，考虑潜伏期具有传染性
from scipy.integrate import odeint  # 导入 scipy.integrate 模块
import numpy as np  # 导入 numpy包
import matplotlib.pyplot as plt  # 导入 matplotlib包


def dySEIR(y, t, lamda, delta, mu):  # SEIR 模型，导数函数
    s, e, i = y
    ds_dt = - lamda * s * i  # ds/dt = -lamda*s*i
    de_dt = lamda * s * i - delta * e  # de/dt = lamda*s*i - delta*e
    di_dt = delta * e - mu * i  # di/dt = delta*e - mu*i
    return np.array([ds_dt, de_dt, di_dt])


def dySEIR2(y, t, lamda, lam2, delta, mu):  # SEIR2 模型，导数函数
    s, e, i = y
    ds_dt = - lamda * s * i - lam2 * s * e  # ds/dt = -lamda*s*i - lam2*s*e
    de_dt = lamda * s * i + lam2 * s * e - delta * e  # de/dt = lamda*s*i - delta*e
    di_dt = delta * e - mu * i  # di/dt = delta*e - mu*i
    return np.array([ds_dt, de_dt, di_dt])


# 设置模型参数
number = 1e5  # 总人数
lamda = 1.0  # 日接触率, 患病者每天有效接触的易感者的平均人数
lam2 = 0.25  # 日接触率2, 潜伏者每天有效接触的易感者的平均人数
delta = 0.05  # 日发病率，每天发病成为患病者的潜伏者占潜伏者总数的比例
mu = 0.05  # 日治愈率, 每天治愈的患病者人数占患病者总数的比例
sigma = lamda / mu  # 传染期接触数
fsig = 1 - 1 / sigma
tEnd = 200  # 预测日期长度
t = np.arange(0.0, tEnd, 1)  # (start,stop,step)
i0 = 1e-3  # 患病者比例的初值
e0 = 0  # 潜伏者比例的初值
s0 = 1 - i0  # 易感者比例的初值
Y0 = (s0, e0, i0)  # 微分方程组的初值

# odeint 数值解，求解微分方程初值问题
ySEIR = odeint(dySEIR, Y0, t, args=(lamda, delta, mu))  # SEIR 模型
ySEIR2 = odeint(dySEIR2, Y0, t, args=(lamda, lam2, delta, mu))  # SEIR2 模型

# 输出绘图
print("lamda={}\tmu={}\tsigma={}\t(1-1/sig)={}".format(lamda, mu, sigma, fsig))
plt.title("Comparison between SEIR and improved SEIR model")
plt.xlabel('t')
plt.axis((0, tEnd, -0.1, 1.1))

plt.plot(t, ySEIR2[:, 0], '-g', label='s(t)-iSEIR')  # 易感者比例
plt.plot(t, ySEIR2[:, 1], '-b', label='e(t)-iSEIR')  # 潜伏者比例
plt.plot(t, ySEIR2[:, 2], '-m', label='i(t)-iSEIR')  # 患病者比例
# plt.plot(t, 1-ySEIR2[:,0]-ySEIR2[:,1]-ySEIR2[:,2], '-b', label='r(t)-iSEIR')
plt.plot(t, ySEIR[:, 0], '--g', label='s(t)-SEIR')
plt.plot(t, ySEIR[:, 1], '--b', label='e(t)-SEIR')
plt.plot(t, ySEIR[:, 2], '--m', label='i(t)-SEIR')
# plt.plot(t, 1-ySEIR[:,0]-ySEIR[:,1]-ySEIR[:,2], '--m', label='r(t)-SEIR')
plt.legend(loc='upper right')
plt.show()
