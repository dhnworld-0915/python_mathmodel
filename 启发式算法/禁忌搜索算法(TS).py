# coding:gbk
import random
import math
import matplotlib.pyplot as plt

global m, best, tl  # m 城市个数 best全局最优   tl初始禁忌长度
global time, spe  # time 迭代次数, spe特赦值
best = 10000.0
m = 14
tl = 8
spe = 5
time = 100
tabu = [[0] * m for i in range(m)]  # 禁忌表
best_way = [0] * m
now_way = [0] * m  # best_way 最优解  now_way当前解
dis = [[0] * m for i in range(m)]  # 两点距离


class no:  # 该类表示每个点的坐标
    def __init__(self, x, y):
        self.x = x
        self.y = y


p = []


def draw(t):  # 该函数用于描绘路线图
    x = [0] * (m + 1)
    y = [0] * (m + 1)
    for i in range(m):
        x[i] = p[t[i]].x
        y[i] = p[t[i]].y
    x[m] = p[t[0]].x
    y[m] = p[t[0]].y
    plt.plot(x, y, color='r', marker='*')
    plt.show()


def mycol():  # 城市坐标输入
    p.append(no(16, 96))
    p.append(no(16, 94))
    p.append(no(20, 92))
    p.append(no(22, 93))
    p.append(no(25, 97))
    p.append(no(22, 96))
    p.append(no(20, 97))
    p.append(no(17, 96))
    p.append(no(16, 97))
    p.append(no(14, 98))
    p.append(no(17, 97))
    p.append(no(21, 95))
    p.append(no(19, 97))
    p.append(no(20, 94))


def get_dis(a, b):  # 返回a，b两城市的距离
    return math.sqrt((p[a].x - p[b].x) * (p[a].x - p[b].x) + (p[a].y - p[b].y) * (p[a].y - p[b].y))


def get_value(t):  # 计算解t的路线长度
    ans = 0.0
    for i in range(1, m):
        ans += dis[t[i]][t[i - 1]]
    ans += dis[t[0]][t[m - 1]]
    return ans


def cop(a, b):  # 把b数组的值赋值a数组
    for i in range(m):
        a[i] = b[i]


def rand(g):  # 随机生成初始解
    vis = [0] * m
    for i in range(m):
        vis[i] = 0
    on = 0
    while on < m:
        te = random.randint(0, m - 1)
        if vis[te] == 0:
            vis[te] = 1
            g[on] = te
            on += 1


def init():  # 初始化函数
    global best
    for i in range(m):
        for j in range(m):
            tabu[i][j] = 0  # 初始化禁忌表
            dis[i][j] = get_dis(i, j)  # 计算两点距离
    rand(now_way)  # 生成初始解作为当前解
    now = get_value(now_way)
    cop(best_way, now_way)
    best = now


def slove():  # 迭代函数
    global best, now
    temp = [0] * m  # 中间变量记录交换结果
    a = 0
    b = 0  # 记录交换城市下标
    ob_way = [0] * m
    cop(ob_way, now_way)
    ob_value = get_value(now_way)  # 暂存邻域最优解
    for i in range(1, m):  # 搜索所有邻域
        for j in range(1, m):
            if (i + j) >= m:
                break
            if i == j:
                continue
            cop(temp, now_way)
            temp[i], temp[i + j] = temp[i + j], temp[i]  # 交换
            value = get_value(temp)
            if value <= best and tabu[i][i + j] < spe:  # 如果优于全局最优且禁忌长度小于特赦值
                cop(best_way, temp)
                best = value
                a = i
                b = i + j  # 更新全局最优且接受新解
                cop(ob_way, temp)
                ob_value = value
            elif tabu[i][i + j] == 0 and value < ob_value:  # 如果优于邻域中的最优解则
                cop(ob_way, temp)
                ob_value = value
                a = i
                b = i + j  # 接受新解

    cop(now_way, ob_way)  # 更新当前解
    for i in range(m):  # 更新禁忌表
        for j in range(m):
            if tabu[i][j] > 0: tabu[i][j] -= 1
    tabu[a][b] = tl  # 重置a，b两个交换城市的禁忌值


# *************************主函数*************************

mycol()  # 数据输入
init()  # 数据初始化

for i in range(time):  # 控制迭代次数
    slove()
print("路线总长度：", round(best, 3))  # 打印最优解距离保留三位小数
draw(best_way)  # 画图描绘路线
print("具体路线：", best_way)  # 打印路线，以序列表示
