'''
（1）问题定义，确定决策变量、目标函数和约束条件；
（2）模型构建，由问题描述建立数学方程，并转化为标准形式的数学模型；
（3）模型求解，用标准模型的优化算法对模型求解，得到优化结果；
'''

import pulp

# 1.线性规划入门
# 定义一个规划问题
MyProbLP = pulp.LpProblem("LPProbDemo1", sense=pulp.LpMaximize)
# 参数 sense 用来指定求最小值/最大值问题，可选参数值：LpMinimize、LpMaximize 。
x1 = pulp.LpVariable('x1', lowBound=0, upBound=7, cat='Continuous')
x2 = pulp.LpVariable('x2', lowBound=0, upBound=7, cat='Continuous')
x3 = pulp.LpVariable('x3', lowBound=0, upBound=7, cat='Continuous')
# 参数 cat 用来设定变量类型，可选参数值：‘Continuous’ 表示连续变量（默认值）、’ Integer ’ 表示离散变量（用于整数规划问题）、’ Binary ’ 表示0/1变量（用于0/1规划问题）。
MyProbLP += 2 * x1 + 3 * x2 - 5 * x3  # 设置目标函数
MyProbLP += (2 * x1 - 5 * x2 + x3 >= 10)  # 不等式约束
MyProbLP += (x1 + 3 * x2 + x3 <= 12)  # 不等式约束
MyProbLP += (x1 + x2 + x3 == 7)  # 等式约束
MyProbLP.solve() # 求解
print("Status:", pulp.LpStatus[MyProbLP.status])  # 输出求解状态
for v in MyProbLP.variables():
    print(v.name, "=", v.varValue)  # 输出每个变量的最优值
print("F(x) = ", pulp.value(MyProbLP.objective))  # 输出最优解的目标函数值

# 2.线性规划进阶：使用 dict 定义决策变量和约束条件
import pulp

# 1. 建立问题
AlloyModel = pulp.LpProblem("钢材生产问题", pulp.LpMinimize)
# 2. 建立变量
material = ['废料1', '废料2', '废料3', '废料4', '镍', '铬', '钼']
mass = pulp.LpVariable.dicts("原料", material, lowBound=0, cat='Continuous')
# 3. 设置目标函数
cost = {
    '废料1': 16,
    '废料2': 10,
    '废料3': 8,
    '废料4': 9,
    '镍': 48,
    '铬': 60,
    '钼': 53}
AlloyModel += pulp.lpSum([cost[item] * mass[item] for item in material]), "总生产成本"
# # 4. 施加约束
carbonPercent = {
    '废料1': 0.8,
    '废料2': 0.7,
    '废料3': 0.85,
    '废料4': 0.4,
    '镍': 0,
    '铬': 0,
    '钼': 0}
NiPercent = {
    '废料1': 18,
    '废料2': 3.2,
    '废料3': 0,
    '废料4': 0,
    '镍': 100,
    '铬': 0,
    '钼': 0}
CrPercent = {
    '废料1': 12,
    '废料2': 1.1,
    '废料3': 0,
    '废料4': 0,
    '镍': 0,
    '铬': 100,
    '钼': 0}
MoPercent = {
    '废料1': 0,
    '废料2': 0.1,
    '废料3': 0,
    '废料4': 0,
    '镍': 0,
    '铬': 0,
    '钼': 100}
AlloyModel += pulp.lpSum([mass[item] for item in material]) == 1000, "质量约束"
AlloyModel += pulp.lpSum([carbonPercent[item] * mass[item] for item in material]) >= 0.65 * 1000, "碳最小占比"
AlloyModel += pulp.lpSum([carbonPercent[item] * mass[item] for item in material]) <= 0.75 * 1000, "碳最大占比"
AlloyModel += pulp.lpSum([NiPercent[item] * mass[item] for item in material]) >= 3.0 * 1000, "镍最小占比"
AlloyModel += pulp.lpSum([NiPercent[item] * mass[item] for item in material]) <= 3.5 * 1000, "镍最大占比"
AlloyModel += pulp.lpSum([CrPercent[item] * mass[item] for item in material]) >= 1.0 * 1000, "铬最小占比"
AlloyModel += pulp.lpSum([CrPercent[item] * mass[item] for item in material]) <= 1.2 * 1000, "铬最大占比"
AlloyModel += pulp.lpSum([MoPercent[item] * mass[item] for item in material]) >= 1.1 * 1000, "钼最小占比"
AlloyModel += pulp.lpSum([MoPercent[item] * mass[item] for item in material]) <= 1.3 * 1000, "钼最大占比"
AlloyModel += mass['废料1'] <= 75, "废料1可用量"
AlloyModel += mass['废料2'] <= 250, "废料2可用量"
# 5. 求解
AlloyModel.solve()
# 6. 打印结果
print(AlloyModel)  # 输出问题设定参数和条件
print("优化状态:", pulp.LpStatus[AlloyModel.status])
for v in AlloyModel.variables():
    print(v.name, "=", v.varValue)
print("最优总成本 = ", pulp.value(AlloyModel.objective))

# 3.线性规划实例：整数规划

import pulp      # 导入 pulp库
ProbILP = pulp.LpProblem("ProbILP", sense=pulp.LpMaximize)  # 定义问题 1，求最大值
x1 = pulp.LpVariable('x1', lowBound=0, upBound=8, cat='Integer')  # 定义 x1，变量类型：整数
x2 = pulp.LpVariable('x2', lowBound=0, upBound=7, cat='Integer')  # 定义 x2，变量类型：整数
ProbILP += (10 * x1 + 9 * x2)  # 设置目标函数 f(x)
ProbILP += (6 * x1 + 5 * x2 <= 60)  # 不等式约束
ProbILP += (10 * x1 + 20 * x2 <= 150)  # 不等式约束
ProbILP.solve()
print(ProbILP.name)  # 输出求解状态
print("Status:", pulp.LpStatus[ProbILP.status])  # 输出求解状态
for v in ProbILP.variables():
    print(v.name, "=", v.varValue)  # 输出每个变量的最优值
print("F5(x) =", pulp.value(ProbILP.objective))  # 输出最优解的目标函数值
