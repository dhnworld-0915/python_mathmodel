# Create graph
from networkx import DiGraph
import matplotlib.pyplot as plt

G = DiGraph()
for v in [1, 2, 3, 4, 5]:
    G.add_edge("Source", v, cost=10)
    G.add_edge(v, "Sink", cost=10)
G.add_edge(1, 2, cost=10)
G.add_edge(2, 3, cost=10)
G.add_edge(3, 4, cost=15)
G.add_edge(4, 5, cost=10)
from vrpy import VehicleRoutingProblem

prob = VehicleRoutingProblem(G)
prob.num_stops = 3  # 可以去到的站点数
prob.solve()
print(prob.best_routes)  # 查看最佳路径
"""{1: ['Source', 4, 5, 'Sink'], 2: ['Source', 1, 2, 3, 'Sink']}"""
print(prob.best_value)  # 最佳路径总的cost
"""70"""
print(prob.best_routes_cost)  # 每条路径各自对应的cost
"""{1: 30, 2: 40}"""

# 定义需求量
for v in G.nodes():
    if v not in ["Source", "Sink"]:
        G.nodes[v]["demand"] = 5
# 容量约束，通过load_capacity来设置
prob = VehicleRoutingProblem(G)  # 原文档中缺少了这个，会报错，因为修改了G，prob需要重新定义
prob.load_capacity = 10

prob.solve()
print(prob.best_routes)
"""{1: ['Source', 4, 5, 'Sink'], 2: ['Source', 2, 3, 'Sink'], 3: ['Source', 1, 'Sink']}"""
print(prob.best_value)
"""80.0"""
print(prob.best_routes_load)
"""{1: 10, 2: 10, 3: 5}"""

# 总时间约束，通过prob.duration定义总限制时间

for (u, v) in G.edges():
    G.edges[u, v]["time"] = 20
G.edges[4, 5]["time"] = 25
prob = VehicleRoutingProblem(G)
prob.duration = 60
prob.solve()
print(prob.best_value)
"""85"""
print(prob.best_routes)
"""{1: ['Source', 3, 4, 'Sink'],
 2: ['Source', 1, 2, 'Sink'],
 3: ['Source', 5, 'Sink']}"""
print(prob.best_routes_duration)
"""{1: 60, 2: 60, 3: 40}"""

# 时间窗限制：假定需要在给定的时间窗口内为客户提供服务，通过对每个节点设置service_time、upper、lower来定义。

import networkx as nx
from vrpy import VehicleRoutingProblem

# Create graph
G = nx.DiGraph()
for v in [1, 2, 3, 4, 5]:
    G.add_edge("Source", v, cost=10, time=20)
    G.add_edge(v, "Sink", cost=10, time=20)
    G.nodes[v]["demand"] = 5
    G.nodes[v]["upper"] = 100
    G.nodes[v]["lower"] = 5
    G.nodes[v]["service_time"] = 1
G.nodes[2]["upper"] = 20
G.nodes["Sink"]["upper"] = 110
G.nodes["Source"]["upper"] = 100
G.add_edge(1, 2, cost=10, time=20)
G.add_edge(2, 3, cost=10, time=20)
G.add_edge(3, 4, cost=15, time=20)
G.add_edge(4, 5, cost=10, time=25)

# Create vrp
prob = VehicleRoutingProblem(G, num_stops=3, load_capacity=10, duration=64, time_windows=True)

# Solve and display solution
prob.solve()
print(prob.best_routes)
"""{1: ['Source', 1, 'Sink'], 2: ['Source', 4, 'Sink'], 3: ['Source', 5, 'Sink'], 4: ['Source', 2, 3, 'Sink']}"""
print(prob.best_value)
"""90"""

# 周期性的CVRP
# 可以为每个节点定义访问频率，通过periodic和frequency设定。
# 假设计划周期为两天，客户2必须访问两次，其他客户仅访问一次：

G.nodes[2]["frequency"] = 2
prob = VehicleRoutingProblem(G, num_stops=3, load_capacity=10)
prob.periodic = 2
prob.solve()
print(prob.best_routes)
"""{1: ['Source', 1, 'Sink'], 2: ['Source', 2, 'Sink'], 3: ['Source', 2, 3, 'Sink'], 4: ['Source', 4, 5, 'Sink']}"""
print(prob.schedule)
"""{0: [1, 2], 1: [3, 4]}"""

# 混合车辆：可能会需要运行不同类型（容量、行驶成本、固定成本）的车辆

from networkx import DiGraph
from vrpy import VehicleRoutingProblem

G = DiGraph()
for v in [1, 2, 3, 4, 5]:
    G.add_edge("Source", v, cost=[10, 11])  # 10是第一辆车的行驶成本，11是第二辆车的行驶成本，下面的以此类推
    G.add_edge(v, "Sink", cost=[10, 11])
    G.nodes[v]["demand"] = 5
G.add_edge(1, 2, cost=[10, 11])
G.add_edge(2, 3, cost=[10, 11])
G.add_edge(3, 4, cost=[15, 16])
G.add_edge(4, 5, cost=[10, 11])
prob = VehicleRoutingProblem(G, mixed_fleet=True, fixed_cost=[0, 5], load_capacity=[5, 20])
# prob = VehicleRoutingProblem(G, num_stops=5)
prob.solve()
print(prob.best_value)
"""85"""
print(prob.best_routes)
"""{1: ['Source', 1, 'Sink'], 2: ['Source', 2, 3, 4, 5, 'Sink']}"""
print(prob.best_routes_cost)
"""{1: 20, 2: 65}"""
print(prob.best_routes_type)
"""{1: 0, 2: 1}"""
pos = nx.spring_layout(G)
nx.draw_networkx(G, pos, with_labels=True, alpha=0.5)
labels = nx.get_edge_attributes(G, 'weight')  # 获取边缘标签
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)  # 绘制边缘标签
plt.show()