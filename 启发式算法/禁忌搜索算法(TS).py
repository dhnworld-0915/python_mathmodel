# coding:gbk
import random
import math
import matplotlib.pyplot as plt

global m, best, tl  # m ���и��� bestȫ������   tl��ʼ���ɳ���
global time, spe  # time ��������, spe����ֵ
best = 10000.0
m = 14
tl = 8
spe = 5
time = 100
tabu = [[0] * m for i in range(m)]  # ���ɱ�
best_way = [0] * m
now_way = [0] * m  # best_way ���Ž�  now_way��ǰ��
dis = [[0] * m for i in range(m)]  # �������


class no:  # �����ʾÿ���������
    def __init__(self, x, y):
        self.x = x
        self.y = y


p = []


def draw(t):  # �ú����������·��ͼ
    x = [0] * (m + 1)
    y = [0] * (m + 1)
    for i in range(m):
        x[i] = p[t[i]].x
        y[i] = p[t[i]].y
    x[m] = p[t[0]].x
    y[m] = p[t[0]].y
    plt.plot(x, y, color='r', marker='*')
    plt.show()


def mycol():  # ������������
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


def get_dis(a, b):  # ����a��b�����еľ���
    return math.sqrt((p[a].x - p[b].x) * (p[a].x - p[b].x) + (p[a].y - p[b].y) * (p[a].y - p[b].y))


def get_value(t):  # �����t��·�߳���
    ans = 0.0
    for i in range(1, m):
        ans += dis[t[i]][t[i - 1]]
    ans += dis[t[0]][t[m - 1]]
    return ans


def cop(a, b):  # ��b�����ֵ��ֵa����
    for i in range(m):
        a[i] = b[i]


def rand(g):  # ������ɳ�ʼ��
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


def init():  # ��ʼ������
    global best
    for i in range(m):
        for j in range(m):
            tabu[i][j] = 0  # ��ʼ�����ɱ�
            dis[i][j] = get_dis(i, j)  # �����������
    rand(now_way)  # ���ɳ�ʼ����Ϊ��ǰ��
    now = get_value(now_way)
    cop(best_way, now_way)
    best = now


def slove():  # ��������
    global best, now
    temp = [0] * m  # �м������¼�������
    a = 0
    b = 0  # ��¼���������±�
    ob_way = [0] * m
    cop(ob_way, now_way)
    ob_value = get_value(now_way)  # �ݴ��������Ž�
    for i in range(1, m):  # ������������
        for j in range(1, m):
            if (i + j) >= m:
                break
            if i == j:
                continue
            cop(temp, now_way)
            temp[i], temp[i + j] = temp[i + j], temp[i]  # ����
            value = get_value(temp)
            if value <= best and tabu[i][i + j] < spe:  # �������ȫ�������ҽ��ɳ���С������ֵ
                cop(best_way, temp)
                best = value
                a = i
                b = i + j  # ����ȫ�������ҽ����½�
                cop(ob_way, temp)
                ob_value = value
            elif tabu[i][i + j] == 0 and value < ob_value:  # ������������е����Ž���
                cop(ob_way, temp)
                ob_value = value
                a = i
                b = i + j  # �����½�

    cop(now_way, ob_way)  # ���µ�ǰ��
    for i in range(m):  # ���½��ɱ�
        for j in range(m):
            if tabu[i][j] > 0: tabu[i][j] -= 1
    tabu[a][b] = tl  # ����a��b�����������еĽ���ֵ


# *************************������*************************

mycol()  # ��������
init()  # ���ݳ�ʼ��

for i in range(time):  # ���Ƶ�������
    slove()
print("·���ܳ��ȣ�", round(best, 3))  # ��ӡ���Ž���뱣����λС��
draw(best_way)  # ��ͼ���·��
print("����·�ߣ�", best_way)  # ��ӡ·�ߣ������б�ʾ
