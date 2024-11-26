import torch
import networkx as nx
import matplotlib.pyplot as plt

# 创建图
G = nx.Graph()

# 节点特征
node_features = torch.tensor([[1.], [0.], [-1.], [0.], [0.], [0.], [0.], [0.], [-1.]])
node_colors = node_features.flatten().numpy()  # 根据节点特征设置颜色

# 添加节点和边
G.add_nodes_from(range(9))
edges = [
    (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8),
    (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8),
    (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8),
    (3, 4), (3, 5), (3, 6), (3, 7), (3, 8),
    (4, 5), (4, 6), (4, 7), (4, 8),
    (5, 6), (5, 7), (5, 8),
    (6, 7), (6, 8),
    (7, 8)
]
G.add_edges_from(edges)

# 绘制图
plt.figure(figsize=(8, 6))
pos = nx.spring_layout(G)  # 使用 spring 布局进行节点排列
nx.draw(G, pos, with_labels=True, node_color=node_colors, cmap=plt.cm.coolwarm, node_size=500, font_size=12, font_color="white", width=2)
plt.title("Quantum Circuit Graph with Node Features")
plt.show()
