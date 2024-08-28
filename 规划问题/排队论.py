"""
排队模型 - 港口系统
"""

import random
import simpy

RANDOM_SEED = 42
NEW_SHIPS = 5  # 总货船数
ARRIVED_INTERVAL = (15, 146)  # 到来时间间隔
UPLOAD_TIME = (45, 91)  # 卸货时长

wait_times = []
upload_times = []


def my_print(*x):
    pass
    # print(*x)


def source(env, number, interval, counter):
    """进程用于生成船"""
    arrived_inter_time = [random.randint(*interval) for i in range(number)]
    arrived_time = [sum(arrived_inter_time[:i + 1]) for i in range(len(arrived_inter_time))]
    my_print("到达时间", arrived_time)
    for i in range(number):
        c = ship(env, ' ' * i * 4 + 'SHIP%02d' % i, counter, arrived_time[i])
        env.process(c)
    yield env.timeout(0)


def ship(env, name, docker, arrived_time):
    yield env.timeout(arrived_time)  # 到达船坞
    my_print(name, '到达时间', env.now)
    with docker.request() as req:  # 寻求进入
        yield req
        my_print(name, '进入时间', env.now)
        unload_time = random.randint(*UPLOAD_TIME)
        wait_time = env.now - arrived_time
        yield env.timeout(unload_time)
        my_print(name, '出港时间', env.now)
        wait_times.append(wait_time / NEW_SHIPS)
        upload_times.append(unload_time / NEW_SHIPS)


ans = []
for i in range(10000):
    wait_times.clear()
    upload_times.clear()
    # Setup and start the simulation
    # random.seed(RANDOM_SEED)
    env = simpy.Environment()

    # Start processes and run
    docker = simpy.Resource(env, capacity=1)
    env.process(source(env, NEW_SHIPS, ARRIVED_INTERVAL, docker))
    env.run()

    ans.append(sum(wait_times))
print("每个船的平均等待时间", sum(ans) / len(ans))

'''
基础知识:
1. random.expovariate(miu) 生成均值为 1/miu 的指数分布的随机数 
2. 泊松过程的强度参数的意义：如果泊松过程的强度参数为 lambda，则在单位时间上新增一次的概率为 lambda，lambda 越大事件越可能发生
3. 泊松事件的事件间隔彼此独立且服从参数为 lambda 的指数分布
4. ρ = λ/μ 
5. 平均等待时间 = ρ/(μ-λ)
6. 平均队列长度（包含正在被服务的人） = λ/(μ-λ)
'''

'''
实现的细节:
1. 统计函数没有将仿真结束时没有被服务完的人算入
'''

import simpy
import random
from time import time
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

## 随机种子
randomSeed = time()  # time()

## 指数分布的均值
miuService = 1  # 单位时间平均离开 1 个人
lambdaReachInterval = 0.5  # 单位时间平均来 0.5 个人

## 服务台的数目
numService = 1

## 仿真程序运行的时间 min
Until = 100

## 系统容量
systemCapacity = None  # None 表示无容量限制 max(10,numService)

## 最大等待时间 超过这个事件之后顾客会离开队伍
maxWaiteTime = Until

## 初始队列长度
initLen = 1


## 客户类
class Customer(object):
    def __init__(self, index_, startTime_, queueLenStart_, reachInterval_):
        self.index = index_  # 第几个来到队列中来的
        self.startTime = startTime_  # 开始时间
        self.getServedTime = None  # 开始被服务的时间
        self.endTime = None  # 结束时间
        self.queueLenStart = queueLenStart_  # 开始排队时队列长度
        self.queueLenEnd = None  # 结束排队时队列长度
        self.reachInterval = reachInterval_  # 空闲了多长时间本 customer 才到达


## 顾客列表
customerList = []

## 当前队列长度
queueLen = 0


class System(object):
    def __init__(self, env, numService, miuService_):
        self.env = env
        self.service = simpy.Resource(env, numService)
        self.miuService = miuService_

    def beingServed(self):
        # 服务事件为均值为 miuService 的指数分布
        yield self.env.timeout(random.expovariate(self.miuService))


def inoutQueue(env, moviegoer, sys):
    # 等待被服务
    with sys.service.request() as request:  # 观众向收银员请求购票
        yield request | env.timeout(maxWaiteTime)  # 观众等待收银员完成面的服务，超过最长等待事件maxCashierTime就会离开
        global customerList
        customerList[moviegoer].getServedTime = env.now
        yield env.process(sys.beingServed())

        # 完善统计资料
    global queueLen
    queueLen -= 1
    customerList[moviegoer].endTime = env.now
    customerList[moviegoer].queueLenEnd = queueLen


def runSys(env, numService, miuService):
    sys = System(env, numService, miuService)
    global initLen, customerList
    moviegoer = initLen
    for moviegoer in range(initLen):  # 初始化设置，开始的队伍长度为 initLen
        customerList.append(Customer(moviegoer, env.now, initLen, 0))
        env.process(inoutQueue(env, moviegoer, sys))
    global queueLen
    queueLen = initLen
    while True:
        reachInterval_ = random.expovariate(lambdaReachInterval)
        yield env.timeout(reachInterval_)  # 顾客到达时间满足 lambdaReachInterval 的指数分布
        if systemCapacity == None or queueLen <= systemCapacity:
            moviegoer += 1
            queueLen += 1
            customerList.append(Customer(moviegoer, env.now, queueLen, reachInterval_))
            env.process(inoutQueue(env, moviegoer, sys))


def plotSimRes(customerList):
    # ! 初始设置
    # 用于正常显示中文标签
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 用来正常显示负号
    plt.rcParams['axes.unicode_minus'] = False

    def plotTime_Service(customerList):
        plt.figure(figsize=(14, 7))  # 新建一个画布
        plt.xlabel('时间/min')
        plt.ylabel('用户序列')
        servedUser = 0
        for customer in customerList:
            y = [customer.index] * 2

            # 等待时间
            if customer.endTime == None:
                customer.endTime = Until
                color = 'r'
            else:
                color = 'b'
                servedUser += 1
            x = [customer.startTime, customer.endTime]
            plt.plot(x, y, color=color)

            # 被服务的时间
            if customer.getServedTime != None and customer.endTime != Until:
                color = 'g'
                x = [customer.getServedTime, customer.endTime]
                plt.plot(x, y, color=color)

        plt.title("时间-队列-服务图 服务的用户数：%d" % servedUser)

    def plotQueueLen_time(customerList):
        plt.figure(figsize=(14, 7))  # 新建一个画布
        plt.xlabel('时间/min')
        plt.ylabel('队列长度/人')

        queueLenList = []

        for customer in customerList:
            queueLenList.append([customer.startTime, customer.queueLenStart])
            queueLenList.append([customer.endTime, customer.queueLenEnd])
        queueLenList.sort()

        preTime = 0
        preLen = 0
        integralQueueLen = 0
        maxLen = 0
        global Until
        timeInCount = Until
        for each in queueLenList:
            if each[1] != None:
                x = [each[0]] * 2
                y = [0, each[1]]
                plt.plot(x, y, color='b')
                plt.plot(each[0], each[1], 'bo')
            else:
                timeInCount = preTime
                break  # 没有把仿真结束时未被服务完的人算进来
            integralQueueLen += (each[0] - preTime) * preLen
            preTime = each[0]
            preLen = each[1]
            maxLen = max(maxLen, each[1])

        averageQueueLen = integralQueueLen / timeInCount
        plt.title("时间-队列长度图 平均队列长度：%f" % averageQueueLen)

    def plotWaiteTime_time(customerList):
        plt.figure(figsize=(14, 7))  # 新建一个画布
        plt.xlabel('时间/min')
        plt.ylabel('等待时间/min')

        queueLenList = []
        peopleInCount = 0
        for customer in customerList:
            if customer.getServedTime != None:
                peopleInCount += 1
                queueLenList.append([customer.startTime, customer.getServedTime - customer.startTime])
        queueLenList.sort()

        integralWaiteTime = 0
        maxWaiteTime = 0
        for each in queueLenList:
            x = [each[0]] * 2
            y = [0, each[1]]
            integralWaiteTime += each[1]
            maxWaiteTime = max(maxWaiteTime, each[1])
            plt.plot(x, y, color='b')
            plt.plot(each[0], each[1], 'bo')

        averageWaiteTime = integralWaiteTime / peopleInCount

        plt.title("时间-等待时间图 平均等待时间：%f" % averageWaiteTime)

    def plotWaiteTime_time_QueueLen(customerList):
        fig = plt.figure(figsize=(14, 7))  # 新建一个画布
        ax = fig.gca(projection='3d')
        plt.xlabel('时间/min')
        plt.ylabel('队列长度/人')
        ax.set_zlabel('等待时间/min')
        plt.title("时间-队列长度-等待时间图")

        queueLenList = []  # 格式：时间 队列长度 等待时间

        global Until
        for customer in customerList:
            if customer.getServedTime != None:  # 没有把仿真结束时未被服务完的人算进来
                queueLenList.append(
                    [customer.startTime, customer.queueLenStart, customer.getServedTime - customer.startTime])
        queueLenList.sort(key=lambda x: x[0])

        for each in queueLenList:
            if each[1] != None:
                x = [each[0]] * 2
                y = [each[1]] * 2
                z = [0, each[2]]
                ax.plot(x, y, z, color='b')
                ax.scatter(x[1], y[1], z[1], c='b')

    plotTime_Service(customerList)
    plotQueueLen_time(customerList)
    plotWaiteTime_time(customerList)
    # plotWaiteTime_time_QueueLen(customerList)
    plt.show()


def main():
    random.seed(randomSeed)
    # 运行模拟场景
    env = simpy.Environment()
    env.process(runSys(env, numService, miuService))
    env.run(until=Until)

    # 查看统计结果
    plotSimRes(customerList)


main()

#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Dang

'''
Part1  设置随机值
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

arrivingtime = np.random.uniform(0,10,size = 20)
arrivingtime.sort()
workingtime = np.random.uniform(1,3,size = 20)
# np.random.uniform 随机数：均匀分布的样本值

startingtime = [0 for i in range(20)]
finishtime = [0 for i in range(20)]
waitingtime = [0 for i in range(20)]
emptytime = [0 for i in range(20)]
# 开始时间都是0
print('arrivingtime\n',arrivingtime,'\n')
print('workingtime\n',workingtime,'\n')
print('startingtime\n',startingtime,'\n')
print('finishtime\n',finishtime,'\n')
print('waitingtime\n',waitingtime,'\n')
print('emptytime\n',emptytime,'\n')

'''
Part2  第一人上厕所时间
'''
startingtime[0] = arrivingtime[0]
# 第一个人之前没有人，所以开始时间 = 到达时间
finishtime[0] = startingtime[0] + workingtime[0]
# 第一个人完成时间 = 开始时间 + “工作”时间
waitingtime[0] = startingtime[0]-arrivingtime[0]
# 第一个人不用等待
print(startingtime[0])
print(finishtime[0])
print(waitingtime[0])

'''
Part3  第二人之后
'''
for i in range(1,len(arrivingtime)):
    if finishtime[i-1] > arrivingtime[i]:
        startingtime[i] = finishtime[i-1]
    else:
        startingtime[i] = arrivingtime[i]
        emptytime[i] = arrivingtime[i] - finishtime[i-1]
    # 判断：如果下一个人在上一个人完成之前到达，则 开始时间 = 上一个人完成时间，
    # 否则 开始时间 = 到达时间，且存在空闲时间 = 到达时间 - 上一个人完成时间
    finishtime[i] = startingtime[i] + workingtime[i]
    waitingtime[i] = startingtime[i] - arrivingtime[i]
    print('第%d个人：到达时间 开始时间 “工作”时间 完成时间 等待时间\n' %i,
          arrivingtime[i],
          startingtime[i],
          workingtime[i],
          finishtime[i],
          waitingtime[i],
         '\n')

print('arerage waiting time is %f' %np.mean(waitingtime))

"""
数据统计
"""
sns.set(style = 'ticks',context = "notebook")
fig = plt.figure(figsize = (8,6))
arrivingtime, = plt.plot(arrivingtime,label = 'arrivingtime')
startingtime, = plt.plot(startingtime,label = 'startingtime')
workingtime, = plt.plot(workingtime,label = 'workingtime')
finishtime, = plt.plot(finishtime,label = 'finishtime')
waitingtime, = plt.plot(waitingtime,label = 'waitingtime')

plt.title(("Queuing problem random simulation experiment").title())

plt.xlabel("Arriving Time(min)")
plt.ylabel("Total Time(min)")

plt.legend(handles=[arrivingtime,startingtime,workingtime,finishtime,waitingtime],
           loc = 'upper left')

plt.show()

# -*- coding:utf-8 -*-
# Author:Xiangyang He
# Coding time: 10h

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # 用于合并绘图


# 定义函数
class P():  # 建立要上厕所的人对象
    def __init__(self, Num, A, WK, WT, ET, FS, ST, R):
        self.Num = Num  # 编号
        self.A = A  # 到达时间 时间点
        self.WT = WT  # 等待时间 时间长度
        self.WK = WK  # “工作”时间 时间长度
        self.ET = ET  # 厕所无人空白时间 时间长度
        self.FS = FS  # 完成工作时间 时间点
        self.ST = ST  # 开始工作时间 时间点
        self.R = R  # 剩余时间 时间长度


def toilet_which(toilet):  # 返回厕所队列中等待时间最短的索引
    lt = []
    for i in toilet:
        lt.append(i.R)
    return lt.index(min(lt))


def toilet_minus_test(toilet, m):  #
    h = []
    for i in range(M):
        if toilet[i] != None:
            h.append(toilet[i].R - m)
    if min(h) <= 0:
        return True
    else:
        return False


def toilet_None(toilet):
    return [i for i, x in enumerate(toilet) if x == None]


def toilet_0(toilet, m):
    return [i for i, x in enumerate(toilet) if x != None and x.R <= m]


def Nature_minus(x, y):  # 自然数集中的减法
    if x > y:
        return x - y
    else:
        return 0


#
N = 200  # item数量
M = 12  # 系统中处理item的个数
K = 30

np.random.seed(2333)

WK = np.random.uniform(10, 5, size=N)  # 工作时间随机生成
A = np.random.uniform(0, K, size=N)  # item到达时间随机生成
A.sort()

y = []
for i in range(0, N - 1):
    y.append(A[i + 1] - A[i])
y = np.array(y)

Queue = [P(i, A[i], WK[i], 0, 0, 0, 0, 0) for i in range(N)]  # 初始化Queue

# 对toilet初始化
Queue[0].ST = Queue[0].A
Queue[0].WT = 0
Queue[0].ET = Queue[0].A
Queue[0].R = Queue[0].WK

toilet = [Queue[0]] + [None for i in range(M - 1)]
lt = []  # 等待的队伍
for k in range(1, N):  # item陆续进入处理器与等待队伍
    m = y[k - 1]
    if toilet_minus_test(toilet, m):

        # print("toilet",toilet,end="")
        s = list(set((toilet_None(toilet) + toilet_0(toilet, m))))
        # print("这是关于A的Num",Queue[k].Num,"时刻为",Queue[k].A)

        if len(lt) == 0:
            print("1,1", Queue[k].Num)
            v = s[0]
            for i in range(M):
                if i == v and toilet[v] == None:
                    Queue[k].ET = Queue[k].A
                    Queue[k].ST = Queue[k].A
                    toilet[v] = Queue[k]
                if i == v and toilet[v] != None:
                    Queue[k].ET = m - toilet[v].R
                    Queue[k].ST = Queue[k].A
                    Queue[k].R = Queue[k].WK
                    toilet[v].FS = toilet[v].ST + toilet[v].WK
                    toilet[v].R = 0
                    toilet[v] = Queue[k]
                if i != v:
                    if toilet[i] != None:
                        toilet[i].R = Nature_minus(toilet[i].R, m)
                        if toilet[i].R == 0:
                            toilet[i].FS = toilet[i].ST + toilet[i].WK
                            toilet[i] = None
        else:
            lt.append(Queue[k])
            # print("1,0apend",Queue[k].Num)
            for i in range(M):
                if i in s:
                    if len(lt) > 1:
                        toilet[i].FS = toilet[i].ST + toilet[i].WK
                        # print("前lt",lt[0].Num)
                        r = lt.pop(0)
                        # print("s",s)
                        # print("后lt",lt[0])

                        # print("第{}号厕所的{}完成".format(i,toilet[i].Num))
                        # print("取出",r.Num,"进入{}厕所".format(i))
                        r.ST = toilet[i].FS
                        r.ET = 0
                        r.R = r.WK - (m - toilet[i].R)
                        toilet[i].R = 0
                        toilet[i] = r
                        # if k == 10:
                        # print("ST",toilet[i].ST,toilet[i].Num,k)
                    if len(lt) == 1:
                        toilet[i].FS = toilet[i].ST + toilet[i].WK
                        toilet[i].R = 0
                        e = lt.pop(0)
                        e.ST = e.A
                        e.R = e.WK
                        toilet[i] = e
                    if len(lt) == 0:
                        toilet[i].FS = toilet[i].ST + toilet[i].WK
                        toilet[i].R = 0
                        toilet[i] = None


                else:
                    toilet[i].R = Nature_minus(toilet[i].R, m)


    else:
        B = None in toilet
        # if k ==2 :
        # L = toilet[2]
        # K = B
        # print(toilet[2])
        if B:
            # print("0,1",Queue[k].Num)
            v = toilet_None(toilet)
            for i in range(M):
                if i == v[0]:
                    Queue[k].ST = Queue[k].A
                    Queue[k].ET = 0
                    Queue[k].R = Queue[k].WK
                    toilet[i] = Queue[k]
                    # if k==2:
                    # print("asjdhaskjdhasd")
                if i not in v:
                    toilet[i].R = Nature_minus(toilet[i].R, m)

        else:
            # print("0,0",Queue[k].Num)
            for i in range(M):
                toilet[i].R = Nature_minus(toilet[i].R, m)
            lt.append(Queue[k])
            # print("append",Queue[k].Num)
    # for i in range(M):
    #   if toilet[i] != None:
    # print("Num,",toilet[i].Num,"R",toilet[i].R)
    # print("m",m)
while len(lt) != 0:  # item已经到达，进入等待队伍
    # print("lt的长度",len(lt))
    v = toilet_which(toilet)
    x = toilet[v].R
    for i in range(M):
        if i == v:
            # print(toilet[v].Num,"在",v,"号厕所拉完",end="")
            toilet[v].FS = toilet[v].ST + toilet[v].WK
            toilet[v].R = 0
            r = lt.pop(0)
            # print(r.Num,"进入{}号厕所".format(v))
            r.ST = toilet[v].FS
            r.ET = 0
            r.R = r.WK
            toilet[v] = r
        else:
            toilet[i].R = Nature_minus(toilet[i].R, x)
for i in range(M):  # 处理器中剩下的工作
    toilet[i].FS = toilet[i].ST + toilet[i].WK
    toilet[i].R = 0
    # print(toilet[i].Num,"在",i,"号厕所拉完",end="")

A = []
ST = []
WK = []
FS = []
WT = []
for i in range(N):
    A.append(Queue[i].A)
    ST.append(Queue[i].ST)
    WK.append(Queue[i].WK)
    FS.append(Queue[i].FS)
    WT.append(Queue[i].WT)
A = np.array(A)
ST = np.array(ST)
WK = np.array(WK)
FS = np.array(FS)
WT = np.array(WT)
WT = ST - A

sns.set(style="ticks", context="notebook")
fig = plt.figure(figsize=(8, 6))
arrivingtime, = plt.plot(A, label="arrivingtime")
startingtime, = plt.plot(ST, label='startingtime')
workingtime, = plt.plot(WK, label='workingtime')
finishtime, = plt.plot(FS, label='finishtime')
waitingtime, = plt.plot(WT, label='waitingtime')
plt.title(("Queuing problem random simulation experiment about {} people to {} toilets".format(N, M)).title())
plt.xlabel("Arriving Time(min)")
plt.ylabel("Total Time(min)")

plt.legend(handles=[arrivingtime, startingtime, workingtime, finishtime, waitingtime],
           loc='upper left')

plt.show()
print("{}个厕所{}分钟，对于{}人，平均每人等待{}分钟".format(M, K, N, np.mean(WT)))
