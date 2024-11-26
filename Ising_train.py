import torch
import torch.optim as optim
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
import pickle
import numpy as np

# 定义 GNN 模型
class GNNModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_edges):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.edge_features = nn.Parameter(torch.ones(num_edges, 1))  # 可学习的边特征
        self.pool = global_mean_pool  # 节点特征池化
        self.fc = nn.Linear(out_channels, 1)  

    def forward(self, x, edge_index, edge_attr):
        # 边特征与输入边特征相乘
        edge_attr = edge_attr * self.edge_features.repeat(2, 1)
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = self.pool(x, batch=torch.zeros(x.size(0), dtype=torch.long))  # 全局池化
        out = self.fc(x)  # 输出标量
        return out

# 创建完整图
def create_full_graph(n_qubits, n_measure, measurement_result):
    x = torch.zeros(n_qubits, 1)  # 节点特征初始化
    for i, qubit in enumerate(n_measure):
        x[qubit] = torch.tensor([measurement_result[i]], dtype=torch.float)

    edge_index = []
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            edge_index.append([i, j])
            edge_index.append([j, i])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    edge_attr = torch.ones(edge_index.shape[1], 1)  # 边特征初始化
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr), edge_index.shape[1]

# 训练函数
def train(model, batch, optimizer, criterion):
    total_loss = 0
    for n_measure, measurement_result, hamiltonian, _ in batch:

        data, num_edges = create_full_graph(n_qubits=9, n_measure=n_measure, measurement_result=measurement_result)
        model.train()
        optimizer.zero_grad()
        predicted_output = model(data.x, data.edge_index, data.edge_attr)

        # 目标值：哈密顿量的最小本征值
        eigenvalues, _ = np.linalg.eigh(hamiltonian)
        hamiltonian_target = torch.tensor(eigenvalues[0], dtype=torch.float).view(1, 1)

        # 计算损失
        loss = criterion(predicted_output, hamiltonian_target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(batch)

# 测试函数
def test(model, dataset):
    model.eval()
    sample = dataset[0]  # 使用第一个样本进行测试
    n_measure, measurement_result, hamiltonian, min_eigenvalue = sample

    # 创建图数据
    data, _ = create_full_graph(n_qubits=9, n_measure=n_measure, measurement_result=measurement_result)

    # 模型预测
    with torch.no_grad():
        predicted_output = model(data.x, data.edge_index, data.edge_attr)

    # 打印结果
    print("实际最小本征值 (Ground Truth):", min_eigenvalue)
    print("预测值 (Predicted Value):", predicted_output.item())
    print("\n节点特征 (Node Features):", data.x)
    print("\n边特征 (Learned Edge Features):", model.edge_features)
    print("\n图连接 (Edge Index):", data.edge_index)

    return data, predicted_output.item()

# 加载数据集
def load_dataset(filename='Ising_dataset.pkl'):
    with open(filename, 'rb') as f:
        dataset = pickle.load(f)
    return dataset

# 分批函数
def get_batches(dataset, batch_size):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i:i + batch_size]

# 训练
if __name__ == "__main__":
    epochs = 100
    batch_size = 8  
    learning_rate = 0.001
    dataset_filename = 'Ising_dataset.pkl' 
    dataset = load_dataset(dataset_filename)


    num_edges = 36  
    model = GNNModel(in_channels=1, hidden_channels=64, out_channels=32, num_edges=num_edges)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # 训练循环
    for epoch in range(epochs):
        total_loss = 0
        for batch in get_batches(dataset, batch_size):
            batch_loss = train(model, batch, optimizer, criterion)
            total_loss += batch_loss * len(batch)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataset)}")

    # 测试
    final_graph, prediction = test(model, dataset)