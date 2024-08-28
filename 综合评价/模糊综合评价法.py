import numpy as np

# TODO：输入评判矩阵

eval_mat = np.array([[0.8, 0.15, 0.05, 0, 0],
                     [0.2, 0.6, 0.1, 0.1, 0],
                     [0.5, 0.4, 0.1, 0, 0],
                     [0.1, 0.3, 0.5, 0.05, 0.05],
                     [0.3, 0.5, 0.15, 0.05, 0],
                     [0.2, 0.2, 0.4, 0.1, 0.1],
                     [0.4, 0.4, 0.1, 0.1, 0],
                     [0.1, 0.3, 0.3, 0.2, 0.1],
                     [0.3, 0.2, 0.2, 0.2, 0.1],
                     [0.1, 0.3, 0.5, 0.1, 0],
                     [0.2, 0.3, 0.3, 0.1, 0.1],
                     [0.2, 0.3, 0.35, 0.15, 0],
                     [0.1, 0.3, 0.4, 0.1, 0.1],
                     [0.1, 0.4, 0.3, 0.1, 0.1],
                     [0.3, 0.4, 0.2, 0.1, 0],
                     [0.1, 0.4, 0.3, 0.1, 0.1],
                     [0.2, 0.3, 0.4, 0.1, 0],
                     [0.4, 0.3, 0.2, 0.1, 0]])
print(eval_mat)

m, n = eval_mat.shape

# TODO: 输入每个一级属性下第一个二级属性的下标
separation_points = [0, 4, 9, 14]

# TODO: 输入每个一级属性下各二级属性的权重
w_mat = np.array([[0.2, 0.3, 0.3, 0.2],
                  [0.3, 0.2, 0.1, 0.2, 0.2],
                  [0.1, 0.2, 0.3, 0.2, 0.2],
                  [0.3, 0.2, 0.2, 0.3]], dtype=object)

# TODO: 输入各一级属性的权重
w_vec = np.array([0.4, 0.3, 0.2, 0.1])

# 计算一级评判向量与二级评判矩阵

separation_points.append(m)
eval_mat_second = []
for i in range(len(separation_points) - 1):
    eval_mat_second.append(w_mat[i] @ eval_mat[separation_points[i]:separation_points[i + 1], :])
eval_mat_second = np.array(eval_mat_second)
print(f"The first-level evaluation vectors are\n{eval_mat_second}")

# 计算二级评判向量

eval_vec = w_vec @ eval_mat_second
print(f"The ultimate evaluation vector is\n{eval_vec}")
