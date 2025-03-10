import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from pathlib import Path

def load_correlation_data(data_folder, N):
    """加载关联数据集"""
    inputs, targets = [], []
    folder = Path(data_folder) / 'correlation_dataset'
    for file in folder.glob(f'gibbs_ising_nq{N}_T*_Z.npy'):
        # 解析文件名中的参数
        parts = file.stem.split('_')
        params = {p[0]: float(p[1:]) for p in parts if p.startswith(('h','j','T'))}
        h, j, t = params['h'], params['j'], params['T']
        
        # 加载数据
        corr = np.load(file)
        inputs.append([h, j, t])
        targets.append(corr)
    
    return np.array(inputs), np.array(targets)

def extract_features(correlations):
    """从关联数据中提取特征"""
    # 计算每个样本的特征：均值、斜率、方差、最大距离关联
    features = []
    for corr in correlations:
        # 关联随距离的衰减斜率（线性拟合）
        j_indices = np.arange(1, len(corr)+1)
        slope = np.polyfit(j_indices, corr, 1)[0]
        
        # 其他统计量
        features.append([
            np.mean(corr),    # 平均关联强度
            slope,            # 关联衰减率
            np.var(corr),     # 关联波动性
            corr[0] - corr[-1] # 首尾关联差异
        ])
    return np.array(features)

def detect_phase_transition(inputs, features):
    """非监督相变检测流程"""
    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    
    # 使用PCA降维
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # DBSCAN聚类（自动发现异常区域）
    db = DBSCAN(eps=0.1, min_samples=14) 
    clusters = db.fit_predict(X_pca)
    
    # 可视化
    plt.figure(figsize=(12, 6))
    
    # PCA投影散点图
    plt.subplot(121)
    scatter = plt.scatter(X_pca[:,0], X_pca[:,1], c=clusters, cmap='viridis', alpha=0.6)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('PCA Projection of Correlation Features')
    plt.colorbar(scatter, label='Cluster')
    
    # 参数空间中的相变区域（h/J vs T）
    plt.subplot(122)
    h_vals = inputs[:,0]
    j_vals = inputs[:,1]
    t_vals = inputs[:,2]
    plt.scatter(h_vals/j_vals, t_vals, c=clusters, cmap='viridis', alpha=0.6)
    plt.xlabel('h/J')
    plt.ylabel('Temperature (T)')
    plt.title('Phase Diagram with Detected Transitions')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("figure/phase_transition_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    return clusters

def main():
    fig_dir = Path("figure")
    fig_dir.mkdir(exist_ok=True)
    # 参数设置
    N = 9
    data_folder = "correlation_dataset"
    
    # 加载数据
    inputs, correlations = load_correlation_data(data_folder, N)
    print(f"Loaded {len(inputs)} samples")
    
    # 特征工程
    features = extract_features(correlations)
    
    # 执行相变检测
    clusters = detect_phase_transition(inputs, features)
    
    # 验证：标记已知临界区域（横向伊辛模型h/J=1附近）
    critical_mask = (inputs[:,0]/inputs[:,1] > 0.9) & (inputs[:,0]/inputs[:,1] < 1.1)
    print(f"Detected anomalies near criticality: {np.sum(clusters[critical_mask] == -1)/np.sum(critical_mask):.1%}")

if __name__ == "__main__":
    main()
