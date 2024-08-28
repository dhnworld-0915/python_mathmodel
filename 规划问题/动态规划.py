"""事例一背包问题
问题描述：假设我们有n种类型的物品，分别编号为1, 2...n。其中编号为i的物品价值为vi，它的重量为wi。
为了简化问题，假定价值和重量都是整数值。现在，假设我们有一个背包，它能够承载的重量是Cap。
现在，我们希望往包里装这些物品，使得包里装的物品价值最大化，那么我们该如何来选择装的东西呢？
注意：每种物品只有一件，可以选择放或者不放。初始化数据为：n = 5，w = {2, 2, 6, 5, 4}，v = {6, 3, 5, 4, 6}，Cap = 10"""
import numpy as np
import time

# 行李数n，不超过的重量W，重量列表w和价值列表p
def fun(n, W, w, p):
    a = np.array([[0] * (W + 1)] * (n + 1))
    # 依次计算前i个行李的最大价值，n+1在n的基础上进行
    for i in range(1, n + 1):
        for j in range(1, W + 1):
            if w[i - 1] > j:
                a[i, j] = a[i - 1, j]
            else:
                a[i, j] = max(a[i - 1, j], p[i - 1] + a[i - 1, j - w[i - 1]])  # 2种情况取最大值
    # print(a)
    print('max value is ' + str(a[n, W]))
    findDetail(p, n, a[n, W])


# 找到价值列表中的一个子集，使得其和等于前面求出的最大价值，即为选择方案
def findDetail(p, n, v):
    a = np.array([[True] * (v + 1)] * (n + 1))
    for i in range(0, n + 1):
        a[i][0] = True
    for i in range(1, v + 1):
        a[0][i] = False
    for i in range(1, n + 1):
        for j in range(1, v + 1):
            if p[i - 1] > j:
                a[i, j] = a[i - 1, j]
            else:
                a[i, j] = a[i - 1, j] or a[i - 1, j - p[i - 1]]
    if a[n, v]:
        i = n
        result = []
        while i >= 0:
            if a[i, v] and not a[i - 1, v]:
                result.append(p[i - 1])
                v -= p[i - 1]
            if v == 0:
                break
            i -= 1
        print(result)
    else:
        print('error')


weights = [1, 2, 5, 6, 7, 9]
price = [1, 6, 18, 22, 28, 36]
fun(len(weights), 13, weights, price)

"""示例二钢条切割
钢条切割，已经各长度的钢条和对应的收益，问长度为n的钢条怎么切割收益最大。"""
# 钢条长度与对应的收益
length = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
profit = (1, 5, 8, 9, 10, 17, 17, 20, 24, 30)


# 参数：profit: 收益列表, n: 钢条总长度
def bottom_up_cut_rod(profit, n):
    r = [0]  # 收益列表
    s = [0] * (n + 1)  # 切割方案列表

    for j in range(1, n + 1):
        q = float('-inf')
        # 每次循环求出长度为j的钢条切割最大收益r[j]，s[j]则保存切割方案中最长的那一段长度
        for i in range(1, j + 1):
            if max(q, profit[length.index(i)] + r[j - i]) == profit[length.index(i)] + r[j - i]:  # 元组index从1开始
                s[j] = i  # 如果切割方案为1和2，那么2会覆盖1，即保存最长的一段
            q = max(q, profit[length.index(i)] + r[j - i])

        r.append(q)
        # r[n]保存长度为n钢条最大切割收益
    return r[n], s[n]


# 切割方案
def rod_cut_method(profit, n):
    how = []
    while n != 0:
        t, s = bottom_up_cut_rod(profit, n)
        how.append(s)
        n -= s

    return how


# 输出长度1~10钢条最大收益和最佳切割方案
for i in range(1, 11):
    t1 = time.time()
    money, s = bottom_up_cut_rod(profit, i)
    how = rod_cut_method(profit, i)
    t2 = time.time()
    print('profit of %d is %d. Cost time is %ss.' % (i, money, t2 - t1))
    print('Cut rod method:%s\n' % how)