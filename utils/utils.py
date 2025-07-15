#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
ESA-3D工具函数
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import scipy.stats as stats


def calculate_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """计算回归指标"""
    metrics = {
        'mse': mean_squared_error(targets, predictions),
        'mae': mean_absolute_error(targets, predictions),
        'rmse': np.sqrt(mean_squared_error(targets, predictions)),
        'r2': r2_score(targets, predictions),
    }
    
    # 皮尔逊相关系数
    pearson_r, pearson_p = stats.pearsonr(targets, predictions)
    metrics['pearson_r'] = pearson_r
    metrics['pearson_p'] = pearson_p
    
    # 斯皮尔曼相关系数
    spearman_r, spearman_p = stats.spearmanr(targets, predictions)
    metrics['spearman_r'] = spearman_r
    metrics['spearman_p'] = spearman_p
    
    return metrics


def plot_predictions(
    predictions: np.ndarray,
    targets: np.ndarray,
    title: str = "Predictions vs Targets",
    save_path: Optional[str] = None,
    show_metrics: bool = True
) -> None:
    """绘制预测值与真实值的散点图"""
    
    # 计算指标
    metrics = calculate_metrics(predictions, targets)
    
    # 创建图像
    plt.figure(figsize=(8, 6))
    
    # 散点图
    plt.scatter(targets, predictions, alpha=0.6, s=30)
    
    # 添加对角线
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    
    # 设置标签和标题
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title(title)
    
    # 添加指标文本
    if show_metrics:
        metrics_text = f"R² = {metrics['r2']:.3f}\n"
        metrics_text += f"MAE = {metrics['mae']:.3f}\n"
        metrics_text += f"RMSE = {metrics['rmse']:.3f}\n"
        metrics_text += f"Pearson r = {metrics['pearson_r']:.3f}"
        
        plt.text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    metrics: Optional[Dict[str, List[float]]] = None,
    save_path: Optional[str] = None
) -> None:
    """绘制训练曲线"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 损失曲线
    axes[0, 0].plot(train_losses, label='Train Loss', color='blue')
    axes[0, 0].plot(val_losses, label='Validation Loss', color='red')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 损失曲线（对数坐标）
    axes[0, 1].plot(train_losses, label='Train Loss', color='blue')
    axes[0, 1].plot(val_losses, label='Validation Loss', color='red')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss (log scale)')
    axes[0, 1].set_title('Training and Validation Loss (Log Scale)')
    axes[0, 1].set_yscale('log')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 指标曲线
    if metrics:
        if 'val_r2' in metrics:
            axes[1, 0].plot(metrics['val_r2'], label='R²', color='green')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('R²')
            axes[1, 0].set_title('Validation R²')
            axes[1, 0].grid(True, alpha=0.3)
        
        if 'val_mae' in metrics:
            axes[1, 1].plot(metrics['val_mae'], label='MAE', color='orange')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('MAE')
            axes[1, 1].set_title('Validation MAE')
            axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def analyze_attention_weights(
    model: torch.nn.Module,
    batch_data: Dict[str, torch.Tensor],
    layer_idx: int = 0,
    head_idx: int = 0
) -> np.ndarray:
    """分析注意力权重"""
    
    model.eval()
    
    # 添加hook来捕获注意力权重
    attention_weights = []
    
    def hook_fn(module, input, output):
        if hasattr(module, 'attention_weights'):
            attention_weights.append(module.attention_weights.detach().cpu().numpy())
    
    # 注册hook
    handle = model.encoder.layers[layer_idx].intra_attention.register_forward_hook(hook_fn)
    
    # 前向传播
    with torch.no_grad():
        output = model(**batch_data)
    
    # 移除hook
    handle.remove()
    
    if attention_weights:
        return attention_weights[0][head_idx]  # 返回指定头的注意力权重
    else:
        return None


def visualize_molecular_graph(
    node_coords: np.ndarray,
    edge_index: np.ndarray,
    block_ids: np.ndarray,
    save_path: Optional[str] = None
) -> None:
    """可视化分子图"""
    
    try:
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("需要安装matplotlib的3D模块来可视化分子图")
        return
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 为不同的区块分配颜色
    unique_blocks = np.unique(block_ids)
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_blocks)))
    
    # 绘制原子（节点）
    for i, block_id in enumerate(unique_blocks):
        mask = block_ids == block_id
        coords = node_coords[mask]
        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], 
                  c=[colors[i]], s=100, alpha=0.7, label=f'Block {block_id}')
    
    # 绘制边
    for i in range(edge_index.shape[1]):
        start_idx, end_idx = edge_index[:, i]
        start_pos = node_coords[start_idx]
        end_pos = node_coords[end_idx]
        
        ax.plot([start_pos[0], end_pos[0]], 
                [start_pos[1], end_pos[1]], 
                [start_pos[2], end_pos[2]], 'k-', alpha=0.3, linewidth=0.5)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Molecular Graph Visualization')
    ax.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def print_model_summary(model: torch.nn.Module) -> None:
    """打印模型摘要"""
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("=" * 50)
    print("模型摘要")
    print("=" * 50)
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")
    print(f"模型大小: {total_params * 4 / 1024 / 1024:.2f} MB")
    print("=" * 50)
    
    # 打印各层参数数量
    print("各层参数数量:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  {name}: {param.numel():,}")
    print("=" * 50)


def save_predictions(
    predictions: np.ndarray,
    targets: np.ndarray,
    pdb_ids: List[str],
    save_path: str
) -> None:
    """保存预测结果"""
    
    results = {
        'pdb_ids': pdb_ids,
        'predictions': predictions.tolist(),
        'targets': targets.tolist(),
        'metrics': calculate_metrics(predictions, targets)
    }
    
    import json
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"预测结果已保存到: {save_path}")


def load_predictions(file_path: str) -> Dict:
    """加载预测结果"""
    
    import json
    with open(file_path, 'r') as f:
        results = json.load(f)
    
    return results


def compare_models(
    results_list: List[Dict],
    model_names: List[str],
    save_path: Optional[str] = None
) -> None:
    """比较多个模型的性能"""
    
    metrics_names = ['mse', 'mae', 'rmse', 'r2', 'pearson_r']
    
    # 创建比较表
    comparison_data = []
    for i, (results, name) in enumerate(zip(results_list, model_names)):
        row = [name]
        for metric in metrics_names:
            if metric in results['metrics']:
                row.append(results['metrics'][metric])
            else:
                row.append(np.nan)
        comparison_data.append(row)
    
    # 创建DataFrame并显示
    try:
        import pandas as pd
        df = pd.DataFrame(comparison_data, columns=['Model'] + metrics_names)
        print("模型性能比较:")
        print(df.to_string(index=False, float_format='%.4f'))
        
        if save_path:
            df.to_csv(save_path, index=False)
            print(f"比较结果已保存到: {save_path}")
            
    except ImportError:
        print("需要安装pandas来生成比较表")


def check_equivariance(
    model: torch.nn.Module,
    batch_data: Dict[str, torch.Tensor],
    rotation_angle: float = 90.0,
    translation: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """检查模型的等变性"""
    
    model.eval()
    
    # 原始预测
    with torch.no_grad():
        output1 = model(**batch_data)
    
    # 创建变换矩阵
    angle_rad = np.radians(rotation_angle)
    rotation_matrix = torch.tensor([
        [np.cos(angle_rad), -np.sin(angle_rad), 0],
        [np.sin(angle_rad), np.cos(angle_rad), 0],
        [0, 0, 1]
    ], dtype=torch.float32)
    
    if translation is None:
        translation = torch.tensor([1.0, 2.0, 3.0])
    else:
        translation = torch.tensor(translation, dtype=torch.float32)
    
    # 应用变换
    transformed_coords = torch.matmul(batch_data['node_coords'], rotation_matrix.T) + translation
    
    # 创建变换后的数据
    transformed_data = batch_data.copy()
    transformed_data['node_coords'] = transformed_coords
    
    # 变换后的预测
    with torch.no_grad():
        output2 = model(**transformed_data)
    
    # 计算差异
    diff = torch.abs(output1 - output2)
    
    equivariance_metrics = {
        'max_diff': diff.max().item(),
        'mean_diff': diff.mean().item(),
        'std_diff': diff.std().item(),
        'is_equivariant': diff.max().item() < 1e-3
    }
    
    return equivariance_metrics


if __name__ == "__main__":
    # 示例使用
    
    # 生成示例数据
    np.random.seed(42)
    predictions = np.random.randn(100) * 2 + 5
    targets = predictions + np.random.randn(100) * 0.5
    
    # 计算指标
    metrics = calculate_metrics(predictions, targets)
    print("示例指标:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # 绘制预测图
    plot_predictions(predictions, targets, title="ESA-3D Example Results")
