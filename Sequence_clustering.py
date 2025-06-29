import torch
import esm
import numpy as np
import pandas as pd
from tqdm import tqdm
from umap import UMAP  
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt
# ------------------------------
# 步骤 1: 读取FASTA文件（含错误处理）
# ------------------------------
def read_fasta(file_path, max_length=1024):
    """
    读取FASTA文件，自动处理多行序列和超长截断
    返回头部列表和序列列表
    """
    headers = []
    sequences = []
    current_header = None
    current_seq = []
    
    try:
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    if current_header is not None:  # 保存上一个序列
                        full_seq = "".join(current_seq).replace("*", "")  # 移除终止符
                        sequences.append(full_seq[:max_length])
                    current_header = line[1:]
                    headers.append(current_header)
                    current_seq = []
                else:
                    current_seq.append(line)
            # 处理最后一个序列
            if current_header is not None:
                full_seq = "".join(current_seq).replace("*", "")
                sequences.append(full_seq[:max_length])
        return headers, sequences
    except Exception as e:
        print(f"读取文件错误: {str(e)}")
        raise

# 输入文件路径（修改为你的FASTA文件路径）
fasta_path = "3w_seq.fasta"
try:
    headers, sequences = read_fasta(fasta_path)
    print(f"成功读取 {len(sequences)} 条序列，示例序列长度: {len(sequences[0])}")
except Exception as e:
    print(f"文件读取失败: {e}")
    exit()

# ------------------------------
# 步骤 2: 加载ESM-2模型（含设备检测）
# ------------------------------
try:
    # 选择模型（根据硬件选择）
    # model_name = "esm2_t12_35M_UR50D"  # CPU兼容
    model_name = "esm2_t33_650M_UR50D"   # GPU推荐
    
    model, alphabet = esm.pretrained.load_model_and_alphabet_hub(model_name)
    model.eval()  # 设置为评估模式
    
    # 自动检测设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"使用设备: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
except Exception as e:
    print(f"模型加载失败: {e}")
    exit()

# ------------------------------
# 步骤 3: 生成嵌入（含显存监控）
# ------------------------------
batch_size = 32 if device.type == "cuda" else 8  # 自动调整批大小
batch_converter = alphabet.get_batch_converter()
embeddings = []

try:
    with tqdm(
        total=len(sequences),
        desc="生成嵌入",
        unit="seq",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
    ) as pbar:
        for i in range(0, len(sequences), batch_size):
            # 准备批次数据
            batch_seqs = sequences[i:i+batch_size]
            batch_headers = headers[i:i+batch_size]
            
            # 转换为模型输入
            batch_data = [(h, s) for h, s in zip(batch_headers, batch_seqs)]
            _, _, batch_tokens = batch_converter(batch_data)
            batch_tokens = batch_tokens.to(device)
            
            # 前向传播
            with torch.no_grad():
                if device.type == "cuda":
                    torch.cuda.empty_cache()  # 清理显存碎片
                results = model(batch_tokens, repr_layers=[33])
            
            # 提取[CLS]向量
            cls_embeddings = results["representations"][33][:, 0].cpu().numpy()
            embeddings.append(cls_embeddings)
            
            pbar.update(len(batch_seqs))  # 更新进度条
    
    X = np.concatenate(embeddings, axis=0)
    print(f"嵌入矩阵形状: {X.shape}")  # (30000, 1280)
    
    # 保存嵌入结果（可选）
    np.save("esm2_embeddings.npy", X)
except RuntimeError as e:
    if "CUDA out of memory" in str(e):
        print(f"显存不足！请降低batch_size（当前值: {batch_size}）")
    else:
        print(f"运行时错误: {e}")
    exit()

# ------------------------------
# 步骤 4: 数据预处理（含方差解释分析）
# ------------------------------
try:
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # UMAP降维至2维（更适合可视化）
    print("正在进行UMAP降维...")
    reducer = UMAP(n_components=2, 
                  n_neighbors=30, 
                  min_dist=0.1, 
                  metric='cosine', 
                  random_state=42)
    X_2d = reducer.fit_transform(X_scaled)
    print("降维完成，形状:", X_2d.shape)
except Exception as e:
    print(f"降维错误: {e}")
    exit()

# ------------------------------
# 步骤 5: 聚类（含进度监控）
# ------------------------------
n_clusters = 20
try:
    print(f"\n开始聚类（{n_clusters}个簇）...")
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=1024,
        n_init=10,
        max_iter=100,
        random_state=42,
        verbose=1
    )
    clusters = kmeans.fit_predict(X_2d)
except Exception as e:
    print(f"聚类错误: {e}")
    exit()

# ------------------------------
# 步骤 6: 保存结果（含完整性检查）
# ------------------------------
if len(headers) == len(clusters):
    result_df = pd.DataFrame({
        "SequenceID": headers,
        "Cluster": clusters,
        "UMAP1": X_2d[:, 0],  # 新增坐标列
        "UMAP2": X_2d[:, 1],
        "SequenceLength": [len(s) for s in sequences]
    })
    try:
        result_df.to_csv("esm2_cluster_results.csv", index=False)
        print(f"结果已保存至 esm2_cluster_results.csv")
        
        # 生成统计摘要
        cluster_stats = result_df.groupby("Cluster").agg({
            "SequenceID": "count",
            "SequenceLength": ["min", "max", "mean"]
        })
        print("\n聚类统计摘要:")
        print(cluster_stats)
    except Exception as e:
        print(f"保存结果失败: {e}")
else:
    print("错误: 序列数量与聚类结果不匹配！")
    # 步骤 7: 评估（基于抽样）
# ------------------------------
try:
    if len(X_2d) > 10000:
        sample_size = 5000  # 减少计算量
    else:
        sample_size = len(X_2d)
    
    sample_idx = np.random.choice(len(X_2d), sample_size, replace=False)
    
    from sklearn.metrics import silhouette_score
    score = silhouette_score(X_2d[sample_idx], clusters[sample_idx])
    print(f"\n轮廓系数（基于{sample_size}个样本）: {score:.3f}")
except Exception as e:
    print(f"评估失败: {e}")
# ------------------------------
# 新增步骤 7: 绘制散点图
# ------------------------------
def plot_clusters(result_path):
    """ 绘制带序列长度信息的聚类散点图（修改版）"""
    print("\n正在生成可视化图表...")
    df = pd.read_csv(result_path)
    
    plt.figure(figsize=(16, 12))
    scatter = plt.scatter(
        x=df["UMAP1"],
        y=df["UMAP2"],
        c=df["Cluster"],          # 颜色表示簇
        cmap="tab20",             # 使用20种颜色
        s=df["SequenceLength"]/5, # 大小反映序列长度
        alpha=0.6,
        edgecolors='none'
    )
    
    # 移除坐标轴标签
    plt.xlabel('')
    plt.ylabel('')
    
    # 移除上、右边框
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    # 添加颜色条
    cbar = plt.colorbar(scatter, ticks=range(df['Cluster'].nunique()))
    cbar.set_label("Cluster ID", rotation=270, labelpad=20)
    
    # 创建单行标题并放在图下方
    title_text = f"Protein Clustering Result (n={len(df)}, Clusters: {df['Cluster'].nunique()}, Silhouette: {score:.2f})"
    plt.figtext(0.5, 0.01, title_text, 
                ha="center", 
                fontsize=12,
                bbox={"facecolor":"white", "alpha":0.8, "pad":5})
    
    # 调整布局，为底部标题留出空间
    plt.subplots_adjust(bottom=0.1)
    
    # 保存高清图
    plt.savefig("esm2_cluster_scatter1.png", dpi=300, bbox_inches='tight')
    print("可视化结果已保存至 esm2_cluster_scatter1.png")

# 执行绘图
plot_clusters("esm2_cluster_results.csv")
