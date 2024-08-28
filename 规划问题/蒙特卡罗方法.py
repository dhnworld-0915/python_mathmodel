# 通常蒙特卡罗方法可以粗略地分成两类：
# 一类是所求解的问题本身具有内在的随机性，借助计算机的运算能力可以直接模拟这种随机的过程。例如在核物理研究中，分析中子在反应堆中的传输过程。中子与原子核作用受到量子力学规律的制约，人们只能知道它们相互作用发生的概率，却无法准确获得中子与原子核作用时的位置以及裂变产生的新中子的行进速率和方向。科学家依据其概率进行随机抽样得到裂变位置、速度和方向，这样模拟大量中子的行为后，经过统计就能获得中子传输的范围，作为反应堆设计的依据。
# 另一种类型是所求解问题可以转化为某种随机分布的特征数，比如随机事件出现的概率，或者随机变量的期望值。通过随机抽样的方法，以随机事件出现的频率估计其概率，或者以抽样的数字特征估算随机变量的数字特征，并将其作为问题的解。这种方法多用于求解复杂的多维积分问题。

# 1.函数模拟
import numpy as np
import random
import matplotlib.pyplot as plt

meanX, stdX = 10, 0.3
n = 1000  # 这里的n就是蒙特卡洛模拟的随机数生成器

x = np.random.normal(meanX, stdX, n)  # 使用numpy内置的正态分布函数random.normal(),随机产生x1000次
y = x * 10 + x ** 2 + 5
np.shape(y)  # 用numpy的shape函数可以查看变量的大小,结果显示y为一个1000行的数组(1000,)
meanY = np.mean(y)
stdY = np.std(y)
print('meanY:' + str(meanY))
print('stdY:' + str(stdY))
# 查看一下x和y随机产生了那些数值
fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
axes[0].bar(np.arange(n), x)
axes[1].bar(np.arange(n), y)
axes[0].set_xlabel('n')
axes[0].set_ylabel('x')
axes[1].set_xlabel('n')
axes[1].set_ylabel('y')
plt.show()
countX, binsX, ignoredX = plt.hist(x, 30, density=True)
plt.plot(binsX, 1 / (stdX * np.sqrt(2 * np.pi)) *
         np.exp(- (binsX - meanX) ** 2 / (2 * stdX ** 2)),
         linewidth=2, color='r')
plt.xlabel('x')
plt.show()
countY, binsY, ignoredY = plt.hist(y, 30, density=True)
plt.plot(binsY, 1 / (stdY * np.sqrt(2 * np.pi)) *
         np.exp(- (binsY - meanY) ** 2 / (2 * stdY ** 2)),
         linewidth=2, color='b')
plt.xlabel('y')
plt.show()

# 2.投针实验
from numpy import random as rd
from numpy import pi, sin

L = 1  # 设置针的长度
a = 2  # 设置平行线之间的距离
n = 100000  # 设置单次模拟次数，次数越多，结果越准确
N = 1000  # 设置总模拟次数
x = rd.random(n) * a / 2  # 在0到a/2上均匀取n个数，表示针的中点与最近平行线的距离
Angle = rd.random(n) * pi  # 在0到pi上均匀取n个数，表示针与最近平行线的夹角
result = []  # 初始化空列表，用来存储每次总模拟的结果
for j in range(N):
    m = 0  # 记录针与平行线相交的次数
    p = 0
    for i in range(n):
        if x[i] <= L / 2 * sin(Angle[i]):  # 判断针与线是否相交
            m += 1
    p = m / n
    Single_Cal_PI = (2 * L) / (a * p)  # 估计pi值
    result.append(Single_Cal_PI)  # 存储进result中
Cal_PI = np.mean(result)  # 求均值
print('经过模拟得到的PI值为：%.8f' % Cal_PI)

# 3.三门问题

from numpy import random as rd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
N = 10000  # 模拟次数
Ch = 0  # 改变选择赢的次数
NoCh = 0  # 不改变选择赢的次数
for i in range(N):
    x = rd.randint(1, 4)  # 从区间[1,4)中取一个整数，表示礼物所在的门号
    y = rd.randint(1, 4)  # 表示初始选择的门号
    if x == y:
        NoCh += 1
    else:
        Ch += 1

print("""共模拟 %d 次
其中改变选择后赢的次数为：Ch=%d
不改变选择后赢的次数为：Noch=%d
""" % (N, Ch, NoCh))

plt.figure()
plt.barh('改变选择', Ch)
plt.barh('不改变选择', NoCh)
plt.title('两种抉择后赢的次数')
plt.show()

# 4.导弹追踪

import numpy as np
from numpy import random as rd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')
# 解决图片的中文乱码问题
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

v = 200  # 任意给定v值
dt = 1e-8  # 定义时间间隔
x = [0, 20]  # 初始化导弹和B船的横坐标
y = [0, 0]  # 初始化两者的纵坐标
t = 0  # 初始化时间
d = 0  # 初始化导弹飞行距离
m = np.sqrt(2) / 2
Distance = np.sqrt((x[1] - x[0]) ** 2 + (y[1] - y[0]) ** 2)
# 导弹与B船的距离

plt.figure()
plt.scatter(x[0], y[0], marker='o', lw=2, color='cornflowerblue')
plt.scatter(x[1], y[1], marker='o', lw=2, color='orange')
plt.grid()
plt.axis((0, 30, 0, 10))
k = 0
while Distance >= 1e-5:  # 击中的临界距离
    t += dt  # 更新时间
    d += 3 * v * dt  # 更新导弹飞行距离
    x[1] = 20 + t * v * m  # 更新B船x坐标
    y[1] = t * v * m  # 更新B船y坐标
    Distance = np.sqrt((x[1] - x[0]) ** 2 + (y[1] - y[0]) ** 2)
    # 更新两者距离
    tan_alpha = (y[1] - y[0]) / (x[1] - x[0])
    cos_alpha = (x[1] - x[0]) / Distance
    sin_alpha = (y[1] - y[0]) / Distance
    x[0] += 3 * v * dt * cos_alpha  # 更新导弹x坐标
    y[0] += 3 * v * dt * sin_alpha  # 更新导弹y坐标
    k += 1
    if k % 2000 == 0:
        plt.plot(x[0], y[0], marker='.', lw=0.5, color='cornflowerblue')
        plt.plot(x[1], y[1], marker='.', lw=0.5, color='orange')
    if d > 50: # 导弹射程为50
        print('导弹没有击中B船！')
        break
    elif d <= 50 and Distance < 1e-5:
        print('导弹飞行%.2f单位距离后击中B船.' % d)
        print('导弹飞行时间为%.2f分钟' % (t * 60))
plt.legend(['导弹运行轨迹', 'B船运行轨迹'])
plt.show()