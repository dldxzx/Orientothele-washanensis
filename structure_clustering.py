import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from Bio import PDB
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import warnings

# 设置随机种子确保结果可复现
np.random.seed(42)
torch.manual_seed(42)
plt.rcParams['figure.figsize'] = (10, 8)
warnings.filterwarnings('ignore')

class ProteinDataset(Dataset):
    """蛋白质结构数据集，用于加载和处理PDB文件"""
    
    def __init__(self, pdb_dir, max_length=1000):
        self.pdb_dir = pdb_dir
        self.pdb_files = [f for f in os.listdir(pdb_dir) if f.endswith('.pdb')]
        self.max_length = max_length
        
    def __len__(self):
        return len(self.pdb_files)
    
    def __getitem__(self, idx):
        pdb_file = os.path.join(self.pdb_dir, self.pdb_files[idx])
        structure = self.load_pdb(pdb_file)
        coords = self.extract_coordinates(structure)
        coords = self.normalize_coordinates(coords)
        coords = self.pad_or_truncate(coords)
        return {
            'coords': torch.tensor(coords, dtype=torch.float32),
            'filename': self.pdb_files[idx]
        }
    
    def load_pdb(self, pdb_file):
        parser = PDB.PDBParser(QUIET=True)
        structure_id = os.path.basename(pdb_file).split('.')[0]
        try:
            structure = parser.get_structure(structure_id, pdb_file)
            return structure
        except Exception as e:
            print(f"Error loading {pdb_file}: {e}")
            return None
    
    def extract_coordinates(self, structure):
        if structure is None:
            return np.zeros((self.max_length, 3))
        
        coords = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    if PDB.is_aa(residue):
                        for atom in residue:
                            if atom.get_name() == 'CA':  # 只提取Cα原子
                                coords.append(atom.get_coord())
        
        if not coords:
            return np.zeros((self.max_length, 3))
            
        return np.array(coords)
    
    def normalize_coordinates(self, coords):
        if len(coords) == 0:
            return coords
            
        centroid = np.mean(coords, axis=0)
        coords = coords - centroid
        scale = np.max(np.linalg.norm(coords, axis=1))
        if scale > 0:
            coords = coords / scale
        return coords
    
    def pad_or_truncate(self, coords):
        if len(coords) >= self.max_length:
            return coords[:self.max_length]
        else:
            padding = np.zeros((self.max_length - len(coords), 3))
            return np.vstack([coords, padding])

class ContrastiveLearner(nn.Module):
    """对比学习模型，用于提取蛋白质结构特征"""
    
    def __init__(self, input_dim=3, hidden_dim=128, projection_dim=64):
        super(ContrastiveLearner, self).__init__()
        
        # 编码器网络
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # 投影头
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim)
        )
        
    def forward(self, x):
        # 交换维度以适应Conv1d的输入要求 (batch, channels, sequence_length)
        x = x.permute(0, 2, 1)
        features = self.encoder(x)
        # 全局平均池化
        features = torch.mean(features, dim=2)
        projections = self.projection_head(features)
        return features, projections

def info_nce_loss(features, temperature=0.1):
    """InfoNCE损失函数，用于对比学习"""
    batch_size = features.shape[0]
    labels = torch.arange(batch_size).to(features.device)
    
    # 计算相似度矩阵
    similarity_matrix = torch.matmul(features, features.T)
    
    # 应用温度缩放
    similarity_matrix = similarity_matrix / temperature
    
    # 计算对比损失
    loss = nn.CrossEntropyLoss()(similarity_matrix, labels)
    return loss

def train_contrastive_model(dataloader, model, optimizer, device, epochs=100):
    """训练对比学习模型"""
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
        
        for i, batch in progress_bar:
            coords = batch['coords'].to(device)
            
            # 应用数据增强（简单的旋转）
            angle = np.random.uniform(0, 2 * np.pi)
            rotation_matrix = torch.tensor([
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1]
            ], dtype=torch.float32).to(device)
            
            coords_augmented = torch.matmul(coords, rotation_matrix)
            
            # 前向传播
            _, projections1 = model(coords)
            _, projections2 = model(coords_augmented)
            
            # 连接两个视图的特征
            all_projections = torch.cat([projections1, projections2], dim=0)
            
            # 计算损失
            loss = info_nce_loss(all_projections)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_description(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/(i+1):.4f}')
        
        print(f'Epoch {epoch+1}/{epochs}, Average Loss: {total_loss/len(dataloader):.4f}')
    
    return model

def extract_features(dataloader, model, device):
    """从训练好的模型中提取特征"""
    model.eval()
    features_list = []
    filenames_list = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Extracting features'):
            coords = batch['coords'].to(device)
            features, _ = model(coords)
            features_list.append(features.cpu().numpy())
            filenames_list.extend(batch['filename'])
    
    features_array = np.vstack(features_list)
    return features_array, filenames_list

def find_optimal_clusters(data, max_clusters=10):
    """使用轮廓系数找到最优聚类数量"""
    silhouette_scores = []
    
    for n_clusters in range(2, max_clusters+1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(data)
        silhouette_avg = silhouette_score(data, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        print(f'For n_clusters = {n_clusters}, the average silhouette_score is: {silhouette_avg:.4f}')
    
    best_n_clusters = np.argmax(silhouette_scores) + 2
    return best_n_clusters

def perform_clustering(features, method='dbscan', n_clusters=10):
    """执行聚类分析"""
    if method == 'kmeans':
        if n_clusters is None:
            n_clusters = find_optimal_clusters(features)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(features)
        return clusters, n_clusters
    
    elif method == 'dbscan':
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        clusters = dbscan.fit_predict(features)
        n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
        return clusters, n_clusters
    
    else:
        raise ValueError(f"Unsupported clustering method: {method}")

def visualize_with_tsne(features, clusters, filenames, output_dir='results'):
    """使用t-SNE进行特征可视化"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 执行t-SNE降维
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_features = tsne.fit_transform(features)
    
    # 创建DataFrame用于绘图
    df = pd.DataFrame({
        'TSNE 1': tsne_features[:, 0],
        'TSNE 2': tsne_features[:, 1],
        'Cluster': clusters,
        'Filename': filenames
    })
    
    # 绘制聚类结果
    plt.figure(figsize=(12, 10))
    scatter = sns.scatterplot(
        x='TSNE 1', y='TSNE 2',
        hue='Cluster',
        palette='viridis',
        s=100,
        data=df,
        legend='full',
        alpha=0.7
    )
    
    # 添加图例和标题
    plt.title('Protein Structure Clusters Visualized by t-SNE', fontsize=16)
    plt.xlabel('TSNE Dimension 1', fontsize=14)
    plt.ylabel('TSNE Dimension 2', fontsize=14)
    plt.legend(title='Cluster', fontsize=12)
    
    # 保存图像
    plt.savefig(os.path.join(output_dir, 'protein_clusters_tsne.png'), dpi=300, bbox_inches='tight')
    
    # 保存聚类结果到CSV
    df.to_csv(os.path.join(output_dir, 'protein_clusters.csv'), index=False)
    
    print(f"聚类结果已保存到 {output_dir} 目录")
    return df

def main(pdb_dir, output_dir='results', batch_size=32, epochs=50):
    """主函数：执行完整的蛋白质结构分析流程"""
    # 检查GPU是否可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 加载数据集
    print(f'加载PDB文件从 {pdb_dir}...')
    dataset = ProteinDataset(pdb_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 初始化模型
    model = ContrastiveLearner().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练对比学习模型
    print('训练对比学习模型...')
    trained_model = train_contrastive_model(dataloader, model, optimizer, device, epochs=epochs)
    
    # 提取特征
    print('从模型中提取特征...')
    features, filenames = extract_features(dataloader, trained_model, device)
    
    # 标准化特征
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # 执行聚类
    print('执行聚类分析...')
    clusters, n_clusters = perform_clustering(scaled_features, method='kmeans')
    
    # 可视化
    print('使用t-SNE进行可视化...')
    results_df = visualize_with_tsne(scaled_features, clusters, filenames, output_dir)
    
    # 打印结果摘要
    print(f'\n分析完成! 共识别出 {n_clusters} 个蛋白质聚类.')
    cluster_counts = results_df['Cluster'].value_counts()
    print('\n聚类分布:')
    for cluster, count in cluster_counts.items():
        print(f'聚类 {cluster}: {count} 个蛋白质')

if __name__ == '__main__':
    # 设置PDB文件目录
    pdb_dir = '/home/user/dwf/3wpdb'  # 请替换为实际的PDB文件目录
    
    # 运行主函数
    main(pdb_dir, output_dir='protein_analysis_results')    