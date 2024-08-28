# 1.随机变量的模拟

from numpy.random import rand
import numpy as np

n = 100000
a = rand(n)
n1 = np.sum(a <= 0.2)
n3 = np.sum(a > 0.5)
n2 = n - n1 - n3
f = np.array([n1, n2, n3]) / n
print(f)

# 2.定积分的计算
from numpy.random import uniform
import numpy as np

N = 10000000
x = uniform(-1, 1, size=N)
y = uniform(-1, 1, N)
z = uniform(0, 1, N)
n = np.sum((x ** 2 + y ** 2 <= 1) & (z >= 0) & (z <= np.sqrt(1 - x ** 2)))
I = n / N * 4
print("I的近似值为：", I)

# 3.在概率计算中的应用

import numpy as np
from scipy.integrate import dblquad

fxy = lambda x, y: 1 / (20000 * np.pi) * np.exp(-(x ** 2 + y ** 2) / 20000)
bdy = lambda x: 80 * np.sqrt(1 - x ** 2 / 120 ** 2)
p1 = dblquad(fxy, -120, 120, lambda x: -bdy(x), bdy)
print("概率的数值解为：", p1)
N = 1000000
mu = [0, 0]
cov = 10000 * np.identity(2)
a = np.random.multivariate_normal(mu, cov, size=N)
n = ((a[:, 0] ** 2 / 120 ** 2 + a[:, 1] ** 2 / 80 ** 2) <= 1).sum()
p2 = n / N
print("概率的近似值为：", p2)

# 4.求全局最优解

import numpy as np
from matplotlib.pyplot import rc, plot, show
from scipy.optimize import fminbound, fmin

rc('font', size=16)
fx = lambda x: (1 - x ** 3) * np.sin(3 * x)
x0 = np.linspace(-2 * np.pi, 2 * np.pi, 100)
y0 = fx(x0)
plot(x0, y0)
show()
xm1 = fminbound(lambda x: -fx(x), -2 * np.pi, 2 * np.pi)
ym1 = fx(xm1)
print(xm1, ym1)
xm2 = fmin(lambda x: -fx(x), -2 * np.pi)
ym2 = fx(xm2)
print(xm2, ym2)
x = np.random.uniform(-2 * np.pi, 2 * np.pi, 100)
y = fx(x)
ym = y.max()
xm = x[y == ym]
print(xm, ym)
