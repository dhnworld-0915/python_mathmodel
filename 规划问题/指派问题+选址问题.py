# 1.指派问题

import pulp  # 导入 pulp 库
import numpy as np


# 主程序
def main():
    # 问题建模：
    """
        决策变量：
            x(i,j) = 0, 第 i 个人不游第 j 种姿势
            x(i,j) = 1, 第 i 个人游第 j 种姿势
            i=1,4, j=1,4
        目标函数：
            min time = sum(sum(c(i,j)*x(i,j))), i=1,4, j=1,4
        约束条件：
            sum(x(i,j),j=1,4)=1, i=1,4
            sum(x(i,j),i=1,4)=1, j=1,4
        变量取值范围：
            x(i,j) = 0,1
    """

    # 游泳比赛的指派问题 (assignment problem)
    # 1.建立优化问题 AssignLP: 求最小值(LpMinimize)
    AssignLP = pulp.LpProblem("Assignment_problem_for_swimming_relay_race", sense=pulp.LpMinimize)  # 定义问题，求最小值
    # 2. 建立变量
    rows = cols = range(0, 4)
    x = pulp.LpVariable.dicts("x", (rows, cols), cat="Binary")
    # 3. 设置目标函数
    scoreM = [[56, 74, 61, 63], [63, 69, 65, 71], [57, 77, 63, 67], [55, 76, 62, 62]]
    AssignLP += pulp.lpSum([[x[row][col] * scoreM[row][col] for row in rows] for col in cols])
    # 4. 施加约束
    for row in rows:
        AssignLP += pulp.lpSum([x[row][col] for col in cols]) == 1  # sum(x(i,j),j=1,4)=1, i=1,4
    for col in cols:
        AssignLP += pulp.lpSum([x[row][col] for row in rows]) == 1  # sum(x(i,j),i=1,4)=1, j=1,4
    # 5. 求解
    AssignLP.solve()
    # 6. 打印结果
    print(AssignLP.name)
    member = ["队员A", "队员B", "队员C", "队员D"]
    style = ["自由泳", "蛙泳", "蝶泳", "仰泳"]
    if pulp.LpStatus[AssignLP.status] == "Optimal":  # 获得最优解
        xValue = [v.varValue for v in AssignLP.variables()]
        # [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        xOpt = np.array(xValue).reshape((4, 4))  # 将 xValue 格式转换为 4x4 矩阵
        print("最佳分配：")
        for row in rows:
            print("{}\t{} 参加项目：{}".format(xOpt[row], member[row], style[np.argmax(xOpt[row])]))
        print("预测最好成绩为：{}".format(pulp.value(AssignLP.objective)))

    return

if __name__ == '__main__':
    main()

# 2.选址问题

import pulp      # 导入 pulp 库

# 主程序
def main():

    # 问题建模：
    """
        决策变量：
            x(j) = 0, 不选择第 j 个消防站
            x(j) = 1, 选择第 j 个消防站, j=1,8
        目标函数：
            min fx = sum(x(j)), j=1,8
        约束条件：
            sum(x(j)*R(i,j),j=1,8) >=1, i=1,8
        变量取值范围：
            x(j) = 0,1
    """

    # 消防站的选址问题 (set covering problem, site selection of fire station)
    # 1.建立优化问题 SetCoverLP: 求最小值(LpMinimize)
    SetCoverLP = pulp.LpProblem("SetCover_problem_for_fire_station", sense=pulp.LpMinimize)  # 定义问题，求最小值
    # 2. 建立变量
    zones = list(range(8))  #  定义各区域
    x = pulp.LpVariable.dicts("zone", zones, cat="Binary")  # 定义 0/1 变量，是否在该区域设消防站
    # 3. 设置目标函数
    SetCoverLP += pulp.lpSum([x[j] for j in range(8)])  # 设置消防站的个数
    # 4. 施加约束
    reachable = [[1, 0, 0, 0, 0, 0, 0, 0],
                 [0, 1, 1, 0, 0, 0, 0, 0],
                 [0, 1, 1, 0, 1, 0, 0, 0],
                 [0, 0, 0, 1, 0, 0, 0, 0],
                 [0, 0, 0, 0, 1, 0, 0, 0],
                 [0, 0, 0, 0, 0, 1, 1, 0],
                 [0, 0, 0, 0, 0, 0, 1, 1],
                 [0, 0, 0, 0, 0, 0, 1, 1]]  # 参数矩阵，第 i 消防站能否在 10分钟内到达第 j 区域
    for i in range(8):
        SetCoverLP += pulp.lpSum([x[j]*reachable[j][i] for j in range(8)]) >= 1

    # 5. 求解
    SetCoverLP.solve()
    # 6. 打印结果
    print(SetCoverLP.name)
    temple = "区域 %(zone)d 的决策是：%(status)s"  # 格式化输出
    if pulp.LpStatus[SetCoverLP.status] == "Optimal":  # 获得最优解
        for i in range(8):
            output = {'zone': i+1,  # 与问题中区域 1~8 一致
                      'status': '建站' if x[i].varValue else '--'}
            print(temple % output)
        print("需要建立 {} 个消防站。".format(pulp.value(SetCoverLP.objective)))

    return

if __name__ == '__main__':
    main()
