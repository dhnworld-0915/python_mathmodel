"""
NetworkX是基于Python语言的图论与复杂网络工具包，用于创建、操作和研究复杂网络的结构、动力学和功能。
NetworkX可以以标准和非标准的数据格式描述图与网络，生成图与网络，分析网络结构，构建网络模型，设计网络算法，绘制网络图形。
NetworkX提供了图形的类、对象、图形生成器、网络生成器、绘图工具，内置了常用的图论和网络分析算法，可以进行图和网络的建模、分析和仿真。
"""

# 1.图的操作

import networkx as nx  # 导入 NetworkX 工具包

# 1.1创建 图
G1 = nx.Graph()  # 创建：空的 无向图
G2 = nx.DiGraph()  # 创建：空的 有向图
G3 = nx.MultiGraph()  # 创建：空的 多图
G4 = nx.MultiDiGraph()  # 创建：空的 有向多图

# 1.2顶点(node)的添加、删除和查看
G1.add_node(1)  # 向 G1 添加顶点 1
G1.add_node(1, name='n1', weight=1.0)  # 添加顶点 1，定义 name, weight 属性
G1.add_node(2, date='May-16')  # 添加顶点 2，定义 time 属性
G1.add_nodes_from([3, 0, 6], dist=1)  # 添加多个顶点：3，0，6
# 查看顶点和顶点属性
print(G1.nodes())  # 查看顶点
# [1, 2, 3, 0, 6]
print(G1.node)  # 查看顶点属性
# {1: {'name': 'n1', 'weight': 1.0}, 2: {'date': 'May-16'}, 3: {'dist': 1}, 0: {'dist': 1}, 6: {'dist': 1}}
H = nx.path_graph(8)  # 创建 路径图 H：由 n个节点、n-1条边连接，节点标签为 0 至 n-1
G1.add_nodes_from(H)  # 由路径图 H 向图 G1 添加顶点 0～9
print(G1.nodes())  # 查看顶点
# [1, 2, 3, 0, 6, 4, 5, 7]  # 顶点列表
G1.add_nodes_from(range(10, 15))  # 向图 G1 添加顶点 10～14
print(G1.nodes())  # 查看顶点
# [1, 2, 3, 0, 6, 4, 5, 7, 10, 11, 12, 13, 14]
# 从图中删除顶点
G1.remove_nodes_from([1, 11, 13, 14])  # 通过顶点标签的 list 删除多个顶点
print(G1.nodes())  # 查看顶点
# [2, 3, 0, 6, 4, 5, 7, 10, 12]  # 顶点列表

# 1.3边(edge)的添加、删除和查看
G1.add_edge(1, 5)  # 向 G1 添加边 1-5，并自动添加图中没有的顶点
G1.add_edge(0, 10, weight=2.7)  # 向 G1 添加边 0-10，并设置属性
G1.add_edges_from([(1, 2, {'weight': 0}), (2, 3, {'color': 'blue'})])  # 向图中添加边，并设置属性
print(G1.nodes())  # 查看顶点
# [2, 3, 0, 6, 4, 5, 7, 10, 12, 1]  # 自动添加了图中没有的顶点 1
G1.add_edges_from([(3, 6), (1, 2), (6, 7), (5, 10), (0, 1)])  # 向图中添加多条边
G1.add_weighted_edges_from([(1, 2, 3.6), [6, 12, 0.5]])  # 向图中添加多条赋权边: (node1,node2,weight)
G1.remove_edge(0, 1)  # 从图中删除边 0-1
# G1.remove_edges_from([(2,3),(1,5),(6,7)])  # 从图中删除多条边
# print(G1.edges(data&result=True))  # 查看所有边的属性
print(G1.edges)  # 查看所有边
# [(2, 1), (2, 3), (3, 6), (0, 10), (6, 7), (6, 12), (5, 1), (5, 10)]
print(G1.get_edge_data(1, 2))  # 查看指定边 1-2 的属性
# {'weight': 3.6}
print(G1[1][2])  # 查看指定边 1-2 的属性
# {'weight': 3.6}

# 1.4查看图、顶点和边的属性
print(G1.nodes)  # 返回所有的顶点 [node1,...]
# [1, 2, 0, 6, 4, 12, 5, 9, 8, 3, 7]
print(G1.edges)  # 返回所有的边 [(node1,node2),...]
# [(1,5), (1,2), (2,8), (2,3), (0,9), (6,5), (6,7), (6,12), (4,3), (4,5), (9,8), (8,7)]
print(G1.degree)  # 返回各顶点的度 [(node1,degree1),...]
# [(1,2), (2,3), (0,1), (6,3), (4,2), (12,1), (5,3), (9,2), (8,3), (3,2), (7,2)]
print(G1.number_of_nodes())  # 返回所有的顶点 [node1,...]
# 11
print(G1.number_of_edges())  # 返回所有的顶点 [node1,...]
# 12
print(G1[2])  # 返回指定顶点相邻的顶点和顶点的属性
# {1: {'weight': 3.6}, 8: {'color': 'blue'}, 3: {}}
print(G1.adj[2])  # 返回指定顶点相邻的顶点和顶点的属性
# {1: {'weight': 3.6}, 8: {'color': 'blue'}, 3: {}}
print(G1[6][12])  # 返回指定边的属性
# {'weight': 0.5}
print(G1.adj[6][12])  # 返回指定边的属性
# {'weight': 0.5}
print(G1.degree(5))  # 返回指定顶点的度
# print('nx.info:',nx.info(G1))  # 返回图的基本信息
print("nodes: ", G1.nodes())  # 输出所有的节点
print("edges: ", G1.edges())  # 输出所有的边
print("number_of_edges: ", G1.number_of_edges())  # 边的条数，只有一条边，就是（2，3）
print('nx.degree:', nx.degree(G1))  # 返回图中各顶点的度
print('nx.density:', nx.degree_histogram(G1))  # 返回图中度的分布
print('nx.pagerank:', nx.pagerank(G1))  # 返回图中各顶点的频率分布

# G.has_node(n)	当图 G 中包括顶点 n 时返回 True
# G.has_edge(u, v)	当图 G 中包括边 (u,v) 时返回 True
# G.number_of_nodes()	返回 图 G 中的顶点的数量
# G.number_of_edges()	返回 图 G 中的边的数量
# G.number_of_selfloops()	返回 图 G 中的自循环边的数量
# G.degree([nbunch, weight])	返回 图 G 中的全部顶点或指定顶点的度
# G.selfloop_edges([data&result, default])	返回 图 G 中的全部的自循环边
# G.subgraph([nodes])	从图 G中抽取顶点[nodes]及对应边构成的子图
# union(G1,G2)	合并图 G1、G2
# nx.info(G)	返回图的基本信息
# nx.degree(G)	返回图中各顶点的度
# nx.degree_histogram(G)	返回图中度的分布
# nx.pagerank(G)	返回图中各顶点的频率分布
# nx.add_star(G,[nodes],**attr)	向图 G 添加星形网络
# nx.add_path(G,[nodes],**attr)	向图 G 添加一条路径
# nx.add_cycle(G,[nodes],**attr)	向图 G 添加闭合路径

# 1.5图的绘制
# 其中，nx.draw() 和 nx.draw_networkx() 是最基本的绘图函数，并可以通过自定义函数属性或其它绘图函数设置不同的绘图要求。常用的属性定义如下：
# ‘node_size’：指定节点的尺寸大小，默认300
# ‘node_color’：指定节点的颜色，默认红色
# ‘node_shape’：节点的形状，默认圆形
# '‘alpha’：透明度，默认1.0，不透明
# ‘width’：边的宽度，默认1.0
# ‘edge_color’：边的颜色，默认黑色
# ‘style’：边的样式，可选 ‘solid’、‘dashed’、‘dotted’、‘dashdot’
# ‘with_labels’：节点是否带标签，默认True
# ‘font_size’：节点标签字体大小，默认12
# ‘font_color’：节点标签字体颜色，默认黑色

import matplotlib.pyplot as plt

G1.clear()  # 清空图G1
nx.add_star(G1, [1, 2, 3, 4, 5], weight=1)  # 添加星形网络：以第一个顶点为中心
# [(1, 2), (1, 3), (1, 4), (1, 5)]
nx.add_path(G1, [5, 6, 8, 9, 10], weight=2)  # 添加路径：顺序连接 n个节点的 n-1条边
# [(5, 6), (6, 8), (8, 9), (9, 10)]
nx.add_cycle(G1, [7, 8, 9, 10, 12], weight=3)  # 添加闭合回路：循环连接 n个节点的 n 条边
# [(7, 8), (7, 12), (8, 9), (9, 10), (10, 12)]
print(G1.nodes)  # 返回所有的顶点 [node1,...]
nx.draw_networkx(G1)
plt.show()

G2 = G1.subgraph([1, 2, 3, 8, 9, 10])
G3 = G1.subgraph([4, 5, 6, 7])
G = nx.union(G2, G3)
print(G.nodes)  # 返回所有的顶点 [node1,...]
# [1, 2, 3, 8, 9, 10, 4, 5, 6, 7]

# 1.6图的分析

# 如果图 G 中的任意两点间相互连通，则 G 是连通图。
G = nx.path_graph(4)
nx.add_path(G, [7, 8, 9])
# 连通子图
listCC = [len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)]
maxCC = max(nx.connected_components(G), key=len)
print('Connected components:{}'.format(listCC))  # 所有连通子图
# Connected components:[4, 3]
print('Largest connected components:{}'.format(maxCC))  # 最大连通子图
# Largest connected components:{0, 1, 2, 3}

# 如果有向图 G 中的任意两点间相互连通，则称 G 是强连通图。
G = nx.path_graph(4, create_using=nx.DiGraph())
nx.add_path(G, [3, 8, 1])
# 找出所有的强连通子图
con = nx.strongly_connected_components(G)
print(type(con), list(con))
# <class 'generator'> [{8, 1, 2, 3}, {0}]

# 如果一个有向图 G 的基图是连通图，则有向图 G 是弱连通图。
G = nx.path_graph(4, create_using=nx.DiGraph())  # 默认生成节点 0,1,2,3 和有向边 0->1,1->2,2->3
nx.add_path(G, [7, 8, 3])  # 生成有向边：7->8->3
con = nx.weakly_connected_components(G)
print(type(con), list(con))
# <class 'generator'> [{0, 1, 2, 3, 7, 8}]

# 2.最短路径

# 2.1Dijkstra算法
# dijkstra_path(G, source, target, weight=‘weight’)
# dijkstra_path_length(G, source, target, weight=‘weight’)
import matplotlib.pyplot as plt  # 导入 Matplotlib 工具包
import networkx as nx  # 导入 NetworkX 工具包

# 无向图的最短路问题（司守奎，数学建模算法与应用，P43，例4.3）
G2 = nx.Graph()  # 创建：空的 无向图
G2.add_weighted_edges_from([(1, 2, 2), (1, 3, 8), (1, 4, 1),
                            (2, 3, 6), (2, 5, 1),
                            (3, 4, 7), (3, 5, 5), (3, 6, 1), (3, 7, 2),
                            (4, 7, 9),
                            (5, 6, 3), (5, 8, 2), (5, 9, 9),
                            (6, 7, 4), (6, 9, 6),
                            (7, 9, 3), (7, 10, 1),
                            (8, 9, 7), (8, 11, 9),
                            (9, 10, 1), (9, 11, 2),
                            (10, 11, 4)])  # 向图中添加多条赋权边: (node1,node2,weight)
# 两个指定顶点之间的最短加权路径
minWPath_v1_v11 = nx.dijkstra_path(G2, source=1, target=11)  # 顶点 0 到 顶点 3 的最短加权路径
print("顶点 v1 到 顶点 v11 的最短加权路径: ", minWPath_v1_v11)
# 两个指定顶点之间的最短加权路径的长度
lMinWPath_v1_v11 = nx.dijkstra_path_length(G2, source=1, target=11)  # 最短加权路径长度
print("顶点 v1 到 顶点 v11 的最短加权路径长度: ", lMinWPath_v1_v11)
pos = nx.spring_layout(G2)  # 用 FR算法排列节点
# 使用nx.draw()绘图时，默认的节点位置可能并不理想，nx.spring_layout() 使用 FR 算法定位节点。
nx.draw(G2, pos, with_labels=True, alpha=0.5)
labels = nx.get_edge_attributes(G2, 'weight')  # 获取边缘标签
nx.draw_networkx_edge_labels(G2, pos, edge_labels=labels)  # 绘制边缘标签
plt.show()

# 2.2Bellman-Ford算法
# bellman_ford_path(G, source, target, weight=‘weight’)
# bellman_ford_path_length(G, source, target, weight=‘weight’)

import pandas as pd
import matplotlib.pyplot as plt  # 导入 Matplotlib 工具包
import networkx as nx  # 导入 NetworkX 工具包

# 城市间机票价格问题（司守奎，数学建模算法与应用，P41，例4.1）
# # 从Pandas数据格式（顶点邻接矩阵）创建 NetworkX 图
# # from_pandas_adjacency(df, create_using=None) # 邻接矩阵，n行*n列，矩阵数据表示权重
dfAdj = pd.DataFrame([[0, 50, 0, 40, 25, 10],  # 0 表示不邻接，
                      [50, 0, 15, 20, 0, 25],
                      [0, 15, 0, 10, 20, 0],
                      [40, 20, 10, 0, 10, 25],
                      [25, 0, 20, 10, 0, 55],
                      [10, 25, 0, 25, 55, 0]])
G1 = nx.from_pandas_adjacency(dfAdj)  # 由 pandas 顶点邻接矩阵 创建 NetworkX 图

# 计算最短路径：注意最短路径与最短加权路径的不同
# 两个指定顶点之间的最短路径
minPath03 = nx.shortest_path(G1, source=0, target=3)  # 顶点 0 到 顶点 3 的最短路径
lMinPath03 = nx.shortest_path_length(G1, source=0, target=3)  # 最短路径长度
print("顶点 0 到 3 的最短路径为：{}，最短路径长度为：{}".format(minPath03, lMinPath03))
# 两个指定顶点之间的最短加权路径
minWPath03 = nx.bellman_ford_path(G1, source=0, target=3)  # 顶点 0 到 顶点 3 的最短加权路径
# 两个指定顶点之间的最短加权路径的长度
lMinWPath03 = nx.bellman_ford_path_length(G1, source=0, target=3)  # 最短加权路径长度
print("顶点 0 到 3 的最短加权路径为：{}，最短加权路径长度为：{}".format(minWPath03, lMinWPath03))

for i in range(1, 6):
    minWPath0 = nx.dijkstra_path(G1, source=0, target=i)  # 顶点 0 到其它顶点的最短加权路径
    lMinPath0 = nx.dijkstra_path_length(G1, source=0, target=i)  # 最短加权路径长度
    print("城市 0 到 城市 {} 机票票价最低的路线为: {}，票价总和为：{}".format(i, minWPath0, lMinPath0))
# nx.draw_networkx(G1) # 默认带边框，顶点标签
nx.draw(G1, with_labels=True, pos=nx.spring_layout(G1))
plt.show()

# 3.条件最短路径

# 3.1图的创建与可视化

import matplotlib.pyplot as plt  # 导入 Matplotlib 工具包
import networkx as nx  # 导入 NetworkX 工具包

# 蚂蚁的最优路径分析（西安邮电大学第12届数学建模竞赛B题）

gAnt = nx.Graph()  # 创建：空的 无向图
gAnt.add_weighted_edges_from([(0, 1, 3), (0, 2, 1), (0, 3, 1),
                              (1, 2, 1), (1, 4, 1), (1, 9, 4),
                              (2, 3, 1), (2, 4, 2), (2, 5, 1),
                              (3, 5, 2), (3, 6, 2), (3, 7, 1),
                              (4, 5, 1), (4, 9, 1),
                              (5, 6, 1), (5, 9, 3), (5, 10, 1), (5, 12, 3),
                              (6, 7, 1), (6, 8, 2), (6, 12, 2), (6, 13, 4), (6, 14, 3),
                              (7, 8, 1),
                              (8, 14, 1), (8, 15, 3),
                              (9, 10, 1), (9, 11, 1),
                              (10, 11, 1), (10, 12, 2),
                              (11, 12, 1), (11, 16, 1),
                              (12, 13, 2), (12, 16, 1),
                              (13, 14, 1), (13, 15, 2), (13, 16, 2), (13, 17, 1),
                              (14, 15, 1),
                              (15, 17, 4),
                              (16, 17, 1)])  # 向图中添加多条赋权边: (node1,node2,weight)
# pos 为字典数据类型，按 node:(x_pos,y_pos) 格式设置节点位置。
pos = {0: (1, 8), 1: (4, 12), 2: (4, 9), 3: (4, 6), 4: (8, 11), 5: (9, 8),  # 指定顶点位置
       6: (11, 6), 7: (8, 4), 8: (12, 2), 9: (12, 13), 10: (15, 11), 11: (18, 13),
       12: (19, 9), 13: (22, 6), 14: (18, 4), 15: (21, 2), 16: (22, 11), 17: (28, 8)}
nx.draw(gAnt, pos, with_labels=True, alpha=0.8)
labels = nx.get_edge_attributes(gAnt, 'weight')
nx.draw_networkx_edge_labels(gAnt, pos, edge_labels=labels, font_color='c')  # 显示权值
nx.draw_networkx_nodes(gAnt, pos, nodelist=[0, 17], node_color='yellow')  # 设置顶点颜色
nx.draw_networkx_nodes(gAnt, pos, nodelist=[7, 12], node_color='lime')  # 设置顶点颜色
nx.draw_networkx_edges(gAnt, pos, edgelist=[(2, 4), (13, 14)], edge_color='lime', width=2.5)  # 设置边的颜色
nx.draw_networkx_edges(gAnt, pos, edgelist=[(11, 12)], edge_color='r', width=2.5)  # 设置边的颜色
plt.show()

# 3.2限制条件：禁止点或禁止边

# 解决方案：从图中删除禁止顶点或禁止边
gAnt.remove_nodes_from([5])  # 通过顶点标签 5 删除顶点
gAnt.remove_edge(13, 17)  # 删除边 (13,17)
minWPath2 = nx.dijkstra_path(gAnt, source=0, target=17)  # 顶点 0 到 顶点 17 的最短加权路径
lMinWPath2 = nx.dijkstra_path_length(gAnt, source=0, target=17)  # 最短加权路径长度
print("\n禁止点或禁止边的约束")
print("S 到 E 的最短加权路径: ", minWPath2)
print("S 到 E 的最短加权路径长度: ", lMinWPath2)

# 3.3限制条件：一个必经点

# 解决方案：分解为两个问题，问题 1 为起点N0至必经点N6，问题 2 为必经点N6至终点N17
minWPath3a = nx.dijkstra_path(gAnt, source=0, target=6)  # N0 到 N6 的最短加权路径
lMinWPath3a = nx.dijkstra_path_length(gAnt, source=0, target=6)  # 最短加权路径长度
minWPath3b = nx.dijkstra_path(gAnt, source=6, target=17)  # N6 到 N17 的最短加权路径
lMinWPath3b = nx.dijkstra_path_length(gAnt, source=6, target=17)  # 最短加权路径长度
minWPath3a.extend(minWPath3b[1:])  # 拼接 minWPath3a、minWPath3b 并去重 N7
print("\n一个必经点的约束")
print("S 到 E 的最短加权路径: ", minWPath3a)
print("S 到 E 的最短加权路径长度: ", lMinWPath3a + lMinWPath3b)

# 3.4限制条件：多个必经点 (N7,N15)

# 解决方案：遍历从起点到终点的简单路径，求满足必经点条件的最短路径
minWPath4 = min([path  # 返回 key 为最小值的 path
                 for path in nx.all_simple_paths(gAnt, 0, 17)  # gAnt 中所有起点为0、终点为17的简单路径
                 if all(n in path for n in (7, 15))],  # 满足路径中包括顶点 N7,N15
                key=lambda x: sum(gAnt.edges[edge]['weight'] for edge in nx.utils.pairwise(x)))  # key 为加权路径长度
lenPath = sum(gAnt.edges[edge]['weight'] for edge in nx.utils.pairwise(minWPath4))  # 求指定路径的加权路径长度
print("\n多个必经点的约束")
print("S 到 E 的最短加权路径: ", minWPath4)
print("S 到 E 的最短加权路径长度: ", lenPath)

# 解决方案：遍历从起点到终点的简单路径，求满足必经点条件的最短路径
lMinWPath5 = minWPath5 = 1e9
for path in nx.all_simple_paths(gAnt, 0, 17):
    if all(n in path for n in (7, 15)):  # 满足路径中包括顶点 N7,N15
        lenPath = sum(gAnt.edges[edge]['weight'] for edge in nx.utils.pairwise(path))
        if lenPath < lMinWPath5:
            lMinWPath5 = lenPath
            minWPath5 = path
print("\n多个必经点的约束")
print("S 到 E 的最短加权路径: ", minWPath5)
print("S 到 E 的最短加权路径长度: ", lMinWPath5)

# 3.5必经边

# 限制条件：必经边 (N2,N4), (N13,N14)，必经点 N7,N12
# 解决方案：遍历从起点到终点的简单路径，求满足必经边条件的最短路径
gAnt.remove_edge(11, 12)  # 禁止边 (11,12)
lMinWPath6 = minWPath6 = 1e9  # 置初值
for path in nx.all_simple_paths(gAnt, 0, 17):  # 所有起点为0、终点为17的简单路径
    if all(n in path for n in (2, 4, 7, 12, 13, 14)):  # 满足路径中包括顶点 N7,N12
        # 检查 (N2,N4)
        p1 = path.index(2)  # N2 的位置
        if path[p1 - 1] != 4 and path[p1 + 1] != 4: continue  # 判断 N2~N4 是否相邻
        # 检查 (N13,N14)
        p2 = path.index(13)  # # N13 的位置
        if path[p2 - 1] != 14 and path[p2 + 1] != 14: continue  # 判断 N13~N14 是否相邻
        lenPath = sum(gAnt.edges[edge]['weight'] for edge in nx.utils.pairwise(path))
        if lenPath < lMinWPath6:
            lMinWPath6 = lenPath
            minWPath6 = path

print("\n多个必经边、必经点的约束")
print("S 到 E 的最短加权路径: ", minWPath6)
print("S 到 E 的最短加权路径长度: ", lMinWPath6)

edgeList = []
for i in range(len(minWPath6) - 1):
    edgeList.append((minWPath6[i], minWPath6[i + 1]))
nx.draw(gAnt, pos, with_labels=True, alpha=0.8)
nx.draw_networkx_edges(gAnt, pos, edgelist=edgeList, edge_color='m', width=4)  # 设置边的颜色
plt.show()

# 4.最小生成树
# minimum_spanning_tree(G, weight=‘weight’, algorithm=‘kruskal’, ignore_nan=False)
# minimum_spanning_edges(G, algorithm=‘kruskal’, weight=‘weight’, keys=True, data&result=True, ignore_nan=False)
# minimum_spanning_tree() 用于计算无向连通图的最小生成树（森林）。
# minimum_spanning_edges() 用于计算无向连通图的最小生成树（森林）的边。
# 对于连通无向图，计算最小生成树；对于非连通无向图，计算最小生成森林。
# minimum_spanning_tree()
# 的返回值是最小生成树，类型为 <class ‘networkx.classes.graph.Graph’ > 。
# minimum_spanning_edges()
# 的返回值是最小生成树的构成边，类型为 <class ‘generator’ > 。

# Demo of minimum spanning tree(MST) with NetworkX

import matplotlib.pyplot as plt  # 导入 Matplotlib 工具包
import networkx as nx  # 导入 NetworkX 工具包

G = nx.Graph()  # 创建：空的 无向图
G.add_weighted_edges_from([(1, 2, 50), (1, 3, 60), (2, 4, 65), (2, 5, 40), (3, 4, 52),
                           (3, 7, 45), (4, 5, 50), (4, 6, 30), (4, 7, 42),
                           (5, 6, 70)])  # 向图中添加多条赋权边: (node1,node2,weight)

T = nx.minimum_spanning_tree(G)  # 返回包括最小生成树的图
print(T.nodes)  # [1, 2, 3, 4, 5, 7, 6]
print(T.edges)  # [(1,2), (2,5), (3,7), (4,6), (4,7), (4,5)]
print(sorted(T.edges))  # [(1,2), (2,5), (3,7), (4,5), (4,6), (4,7)]
print(sorted(T.edges(data=True)))  # data&result=True 表示返回值包括边的权重
# [(1,2,{'weight':50}), (2,5,{'weight':40}), (3,7,{'weight':45}), (4,5,{'weight':50}), (4,6,{'weight':30}), (4,7,{'weight':42})]

mst1 = nx.tree.minimum_spanning_edges(G, algorithm="kruskal")  # 返回值 带权的边
print(list(mst1))
# [(4,6,{'weight':30}), (2,5,{'weight':40}), (4,7,{'weight':42}), (3,7,{'weight':45}), (1,2,{'weight':50}), (4,5,{'weight':50})]
mst2 = nx.tree.minimum_spanning_edges(G, algorithm="prim", data=False)  # data&result=False 表示返回值不带权
print(list(mst2))
# [(1,2), (2,5), (5,4), (4,6), (4,7), (7,3)]

pos = {1: (2.5, 10), 2: (0, 5), 3: (7.5, 10), 4: (5, 5), 5: (2.5, 0), 6: (7.5, 0), 7: (10, 5)}  # 指定顶点位置
nx.draw(G, pos, with_labels=True, alpha=0.8)  # 绘制无向图
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_color='c')  # 显示边的权值
nx.draw_networkx_edges(G, pos, edgelist=T.edges, edge_color='r', width=4)  # 设置指定边的颜色
plt.show()

# 3.5关键路径法
# 关键路径法（Critical path method，CPM）是一种计划管理方法，通过分析项目过程中工序进度安排寻找关键路径，确定最短工期，广泛应用于系统分析和项目管理。

# Demo of critical path method(CPM) with NetworkX

import matplotlib.pyplot as plt  # 导入 Matplotlib 工具包
import networkx as nx  # 导入 NetworkX 工具包

DG = nx.DiGraph()  # 创建：空的 有向图
DG.add_nodes_from(range(1, 8), VE=0, VL=0)
DG.add_weighted_edges_from([(1, 2, 5), (1, 3, 10), (1, 4, 11),
                            (2, 5, 4),
                            (3, 4, 4), (3, 5, 0),
                            (4, 6, 15),
                            (5, 6, 21), (5, 7, 25), (5, 8, 35),
                            (6, 7, 0), (6, 8, 20),
                            (7, 8, 15)])  # 向图中添加多条赋权边: (node1,node2,weight)
lenNodes = len(DG.nodes)  # 顶点数量 YouCans
topoSeq = list(nx.topological_sort(DG))  # 拓扑序列: [1, 3, 4, 2, 5, 7, 6, 8]

# --- 计算各顶点的 VE：事件最早开始时间 ---
VE = [0 for i in range(lenNodes)]  # 初始化 事件最早开始时间
for i in range(lenNodes):
    for e in DG.in_edges(topoSeq[i]):  # 遍历顶点 topoSeq[i] 的 入边
        VEij = DG.nodes[e[0]]["VE"] + DG[e[0]][e[1]]['weight']  # 该路线的最早开始时间
        if VEij > VE[i]: # 该路线所需时间更长
            VE[i] = VEij
    DG.add_node(topoSeq[i], VE=VE[i])  # 顶点（事件）的最早开始时间

# --- 计算各顶点的 VL：事件最晚开始时间 ---
revSeq = list(reversed(topoSeq))  # 翻转拓扑序列，以便从终点倒推计算 VL
VL = [DG.nodes[revSeq[0]]["VE"] for i in range(lenNodes)]  # 初始化 事件最晚开始时间为 VE 最大值
for i in range(lenNodes):
    for e in DG.out_edges(revSeq[i]):  # 遍历顶点 revSeq[i] 的 出边
        VLij = DG.nodes[e[1]]["VL"] - DG[e[0]][e[1]]['weight']  # 该路线的最晚开始时间
        if VLij < VL[i]: # 该路线所需时间更长
            VL[i] = VLij
    DG.add_node(revSeq[i], VL=VL[i])  # 顶点（事件）的最晚开始时间

print("\n顶点（事件）的最早开始时间 VE, 最晚开始时间 VL:")
for n in DG.nodes:  # 遍历有向图的顶点
    print("\t事件 {}:\tVE= {}\tVL= {}".format(n, DG.nodes[n]["VE"], DG.nodes[n]["VL"]))

# --- 计算各条边的 EE, EL：工序最早、最晚开始时间 ---
cpDG = nx.DiGraph()  # 创建空的有向图, 保存关键路径
print("\n边（工序）的最早开始时间 EE, 最晚开始时间 EL:")
for e in DG.edges:  # 遍历有向图的边
    DG[e[0]][e[1]]["EE"] = DG.nodes[e[0]]["VE"]  # 边的头顶点的 VE
    # Wij = DG[e[0]][e[1]]['weight']
    DG[e[0]][e[1]]["EL"] = DG.nodes[e[1]]["VL"] - DG[e[0]][e[1]]['weight']  # 边的尾顶点的 VL 减去边的权值
    if DG[e[0]][e[1]]["EE"] == DG[e[0]][e[1]]["EL"]:  # 如果最早、最晚开工时间相同，则为关键路径上的边
        cpDG.add_edge(e[0], e[1], weight=DG[e[0]][e[1]]['weight'])  # 加入 关键路径
    print("\t工序 {}:\tEE= {}\tEL= {}".format(e, DG[e[0]][e[1]]["EE"], DG[e[0]][e[1]]["EL"]))

lenCP = sum(cpDG[e[0]][e[1]]['weight'] for e in cpDG.edges)
print("\n关键路径:{}".format(cpDG.edges))
print("关键路径长度:{}".format(lenCP))

pos = {1: (0, 4), 2: (5, 8), 3: (5, 4), 4: (5, 0), 5: (10, 8), 6: (10, 0), 7: (15, 4), 8: (20, 4)}  # 指定顶点位置
nx.draw(DG, pos, with_labels=True, alpha=0.8)  # 绘制无向图
labels = nx.get_edge_attributes(DG, 'weight')
nx.draw_networkx_edge_labels(DG, pos, edge_labels=labels, font_color='c')  # 显示边的权值
nx.draw_networkx_edges(DG, pos, edgelist=cpDG.edges, edge_color='r', width=4)  # 设置指定边的颜色
plt.show()