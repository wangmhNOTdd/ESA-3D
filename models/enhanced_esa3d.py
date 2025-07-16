#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
增强的ESA-3D模型
支持多粒度建图和优化的边注意力机制
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch
from torch_scatter import scatter_mean, scatter_sum
from typing import Optional, Tuple, List, Union

from modules.optimized_edge_attention import (
    OptimizedESA3DBlock,
    SparseEquivariantEdgeAttention,
    create_intra_block_mask,
    create_inter_block_mask,
)


class EnhancedEdgeEmbedding(nn.Module):
    """增强的边嵌入层，支持多粒度特征"""
    
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
        granularity: str = 'atom',  # 'atom', 'residue', 'mixed'
        activation: nn.Module = nn.SiLU(),
    ):
        super().__init__()
        
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.granularity = granularity
        
        # 节点特征到边特征的映射
        input_dim = node_dim * 2  # 连接两个节点的特征
        if edge_dim > 0:
            input_dim += edge_dim
        
        # 根据粒度调整网络结构
        if granularity == 'residue':
            # 残基级建图需要更复杂的特征融合
            self.edge_mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim * 3),
                activation,
                nn.Linear(hidden_dim * 3, hidden_dim * 2),
                activation,
                nn.Linear(hidden_dim * 2, hidden_dim),
                activation,
                nn.Linear(hidden_dim, hidden_dim),
            )
        else:
            # 原子级或混合级建图
            self.edge_mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim * 2),
                activation,
                nn.Linear(hidden_dim * 2, hidden_dim),
                activation,
                nn.Linear(hidden_dim, hidden_dim),
            )
        
        # 边的坐标特征：相对位置向量
        self.coord_mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            activation,
            nn.Linear(hidden_dim, 3),
        )
        
    def forward(
        self,
        node_features: torch.Tensor,  # [N, node_dim]
        node_coords: torch.Tensor,    # [N, 3]
        edge_index: torch.Tensor,     # [2, E]
        edge_attr: Optional[torch.Tensor] = None,  # [E, edge_dim]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            node_features: [N, node_dim] 节点特征
            node_coords: [N, 3] 节点坐标
            edge_index: [2, E] 边的索引
            edge_attr: [E, edge_dim] 边属性
        
        Returns:
            edge_features: [E, hidden_dim] 边的不变特征
            edge_coords: [E, 3] 边的等变特征
        """
        src, dst = edge_index
        
        # 节点特征连接
        edge_node_features = torch.cat([node_features[src], node_features[dst]], dim=1)
        
        # 添加边属性
        if edge_attr is not None:
            edge_node_features = torch.cat([edge_node_features, edge_attr], dim=1)
        
        # 生成边的不变特征
        edge_features = self.edge_mlp(edge_node_features)
        
        # 计算相对位置向量
        relative_coords = node_coords[dst] - node_coords[src]  # [E, 3]
        
        # 生成边的等变特征
        edge_coords = self.coord_mlp(relative_coords)
        
        return edge_features, edge_coords


class MultiGranularityGlobalPooling(nn.Module):
    """多粒度全局池化层"""
    
    def __init__(
        self,
        hidden_dim: int,
        num_seeds: int = 32,
        granularity: str = 'atom',
        activation: nn.Module = nn.SiLU(),
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_seeds = num_seeds
        self.granularity = granularity
        
        # 根据粒度调整种子数量
        if granularity == 'residue':
            self.num_seeds = max(16, num_seeds // 2)  # 残基级使用更少的种子
        elif granularity == 'mixed':
            self.num_seeds = num_seeds
        
        # 可学习的种子向量
        self.seed_vectors = nn.Parameter(torch.randn(self.num_seeds, hidden_dim))
        
        # 多头注意力
        self.num_heads = 8
        self.head_dim = hidden_dim // self.num_heads
        
        assert hidden_dim % self.num_heads == 0
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        
        # 最终投影
        self.final_mlp = nn.Sequential(
            nn.Linear(hidden_dim * self.num_seeds, hidden_dim),
            activation,
            nn.Linear(hidden_dim, hidden_dim),
        )
        
    def forward(
        self,
        edge_features: torch.Tensor,  # [E, hidden_dim]
        edge_coords: torch.Tensor,    # [E, 3]
        batch: Optional[torch.Tensor] = None,  # [E] 批次信息
    ) -> torch.Tensor:
        """
        Args:
            edge_features: [E, hidden_dim] 边的不变特征
            edge_coords: [E, 3] 边的等变特征
            batch: [E] 批次信息
        
        Returns:
            graph_repr: [B, hidden_dim] 图的全局表示
        """
        if batch is None:
            batch = torch.zeros(edge_features.shape[0], dtype=torch.long, device=edge_features.device)
        
        batch_size = batch.max().item() + 1
        
        # 为每个批次处理
        graph_reprs = []
        
        for b in range(batch_size):
            mask = batch == b
            if mask.sum() == 0:
                continue
            
            batch_edge_features = edge_features[mask]  # [E_b, hidden_dim]
            batch_edge_coords = edge_coords[mask]      # [E_b, 3]
            
            # 计算种子向量到边特征的注意力
            seeds = self.seed_vectors.unsqueeze(0).expand(1, -1, -1)  # [1, num_seeds, hidden_dim]
            
            # Query, Key, Value
            q = self.q_proj(seeds).view(1, self.num_seeds, self.num_heads, self.head_dim)
            k = self.k_proj(batch_edge_features).view(1, -1, self.num_heads, self.head_dim)
            v = self.v_proj(batch_edge_features).view(1, -1, self.num_heads, self.head_dim)
            
            # 多头注意力
            attention_scores = torch.einsum('bqhd,bkhd->bhqk', q, k) / (self.head_dim ** 0.5)
            attention_weights = F.softmax(attention_scores, dim=-1)
            
            # 加权聚合
            attended_features = torch.einsum('bhqk,bkhd->bqhd', attention_weights, v)
            attended_features = attended_features.view(1, self.num_seeds, self.hidden_dim)
            
            # 输出投影
            graph_repr = self.out_proj(attended_features).squeeze(0)  # [num_seeds, hidden_dim]
            
            # 最终融合
            graph_repr = self.final_mlp(graph_repr.flatten())  # [hidden_dim]
            
            graph_reprs.append(graph_repr)
        
        return torch.stack(graph_reprs)


class EnhancedESA3DEncoder(nn.Module):
    """增强的ESA-3D编码器"""
    
    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 6,
        num_heads: int = 8,
        num_radial: int = 64,
        cutoff: float = 10.0,
        attention_type: str = 'sparse',  # 'sparse', 'block', 'full'
        k_neighbors: int = 32,
        block_size: int = 64,
        activation: nn.Module = nn.SiLU(),
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.attention_type = attention_type
        
        # 构建多层优化的ESA-3D块
        self.layers = nn.ModuleList([
            OptimizedESA3DBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_radial=num_radial,
                cutoff=cutoff,
                attention_type=attention_type,
                k_neighbors=k_neighbors,
                block_size=block_size,
                activation=activation,
                dropout=dropout,
            ) for _ in range(num_layers)
        ])
        
    def forward(
        self,
        edge_features: torch.Tensor,  # [E, hidden_dim]
        edge_coords: torch.Tensor,    # [E, 3]
        edge_index: torch.Tensor,     # [2, E]
        node_coords: torch.Tensor,    # [N, 3]
        block_ids: torch.Tensor,      # [N] 每个节点的区块ID
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            edge_features: [E, hidden_dim] 边的不变特征
            edge_coords: [E, 3] 边的等变特征
            edge_index: [2, E] 边的索引
            node_coords: [N, 3] 节点坐标
            block_ids: [N] 每个节点的区块ID
        
        Returns:
            final_edge_features: [E, hidden_dim] 最终的边不变特征
            final_edge_coords: [E, 3] 最终的边等变特征
        """
        device = edge_features.device
        
        # 创建掩码（根据边数量选择稀疏或密集掩码）
        use_sparse_mask = edge_features.shape[0] > 200
        intra_mask = create_intra_block_mask(edge_index, block_ids, device, use_sparse_mask)
        inter_mask = create_inter_block_mask(edge_index, block_ids, device, use_sparse_mask)
        
        # 逐层处理
        current_edge_features = edge_features
        current_edge_coords = edge_coords
        
        for layer in self.layers:
            current_edge_features, current_edge_coords = layer(
                current_edge_features,
                current_edge_coords,
                edge_index,
                node_coords,
                intra_mask,
                inter_mask,
            )
        
        return current_edge_features, current_edge_coords


class EnhancedESA3DModel(nn.Module):
    """增强的ESA-3D模型 - 支持多粒度和优化的注意力"""
    
    def __init__(
        self,
        node_dim: int = 43,
        edge_dim: int = 16,
        hidden_dim: int = 128,
        num_layers: int = 6,
        num_heads: int = 8,
        num_radial: int = 64,
        cutoff: float = 10.0,
        num_seeds: int = 32,
        output_dim: int = 1,
        granularity: str = 'atom',  # 'atom', 'residue', 'mixed'
        attention_type: str = 'sparse',  # 'sparse', 'block', 'full'
        k_neighbors: int = 32,
        block_size: int = 64,
        activation: nn.Module = nn.SiLU(),
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.granularity = granularity
        self.attention_type = attention_type
        
        # 边嵌入层
        self.edge_embedding = EnhancedEdgeEmbedding(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            granularity=granularity,
            activation=activation,
        )
        
        # ESA-3D编码器
        self.encoder = EnhancedESA3DEncoder(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            num_radial=num_radial,
            cutoff=cutoff,
            attention_type=attention_type,
            k_neighbors=k_neighbors,
            block_size=block_size,
            activation=activation,
            dropout=dropout,
        )
        
        # 全局池化
        self.pooling = MultiGranularityGlobalPooling(
            hidden_dim=hidden_dim,
            num_seeds=num_seeds,
            granularity=granularity,
            activation=activation,
        )
        
        # 输出层
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            activation,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            activation,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )
        
    def forward(self, data):
        """
        Args:
            data: PyG Data对象，包含 x, pos, edge_index, edge_attr, block_ids, batch
        
        Returns:
            output: [B, output_dim] 模型输出
        """
        
        # 边嵌入
        edge_features, edge_coords = self.edge_embedding(
            data.x, data.pos, data.edge_index, data.edge_attr
        )
        
        # ESA-3D编码
        final_edge_features, final_edge_coords = self.encoder(
            edge_features, edge_coords, data.edge_index, data.pos, data.block_ids
        )
        
        # 生成边的批次信息
        if hasattr(data, 'batch') and data.batch is not None:
            edge_batch = data.batch[data.edge_index[0]]
        else:
            edge_batch = None
        
        # 全局池化
        graph_repr = self.pooling(final_edge_features, final_edge_coords, edge_batch)
        
        # 输出预测
        output = self.output_mlp(graph_repr)
        
        return output


def create_enhanced_model_config(
    granularity: str = 'mixed',
    attention_type: str = 'sparse',
    model_size: str = 'medium'  # 'small', 'medium', 'large'
) -> dict:
    """创建增强模型配置"""
    
    # 基础配置
    base_config = {
        "granularity": granularity,
        "attention_type": attention_type,
        "activation": "SiLU",
        "dropout": 0.1,
    }
    
    # 模型大小配置
    if model_size == 'small':
        base_config.update({
            "hidden_dim": 64,
            "num_layers": 4,
            "num_heads": 4,
            "num_radial": 32,
            "num_seeds": 16,
            "k_neighbors": 16,
            "block_size": 32,
        })
    elif model_size == 'medium':
        base_config.update({
            "hidden_dim": 128,
            "num_layers": 6,
            "num_heads": 8,
            "num_radial": 64,
            "num_seeds": 32,
            "k_neighbors": 32,
            "block_size": 64,
        })
    elif model_size == 'large':
        base_config.update({
            "hidden_dim": 256,
            "num_layers": 8,
            "num_heads": 16,
            "num_radial": 128,
            "num_seeds": 64,
            "k_neighbors": 64,
            "block_size": 128,
        })
    
    # 根据粒度调整特征维度
    if granularity == 'atom':
        base_config.update({
            "node_dim": 43,  # element(11) + residue(21) + atom_pos(11)
            "edge_dim": 16,
            "cutoff": 5.0,
        })
    elif granularity == 'residue':
        base_config.update({
            "node_dim": 33,  # residue(21) + size(2) + element_counts(10)
            "edge_dim": 16,
            "cutoff": 10.0,
        })
    elif granularity == 'mixed':
        base_config.update({
            "node_dim": 43,  # element(11) + residue(21) + atom_pos(11)
            "edge_dim": 16,
            "cutoff": 5.0,
        })
    
    return base_config


if __name__ == "__main__":
    # 测试增强的ESA-3D模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建测试数据
    from torch_geometric.data import Data
    
    N = 100  # 节点数
    E = 200  # 边数
    
    data = Data(
        x=torch.randn(N, 43, device=device),
        pos=torch.randn(N, 3, device=device),
        edge_index=torch.randint(0, N, (2, E), device=device),
        edge_attr=torch.randn(E, 16, device=device),
        block_ids=torch.randint(0, 20, (N,), device=device),
        y=torch.randn(1, 1, device=device),
    )
    
    # 测试不同的配置
    for granularity in ['atom', 'residue', 'mixed']:
        for attention_type in ['sparse', 'block']:
            for model_size in ['small', 'medium']:
                print(f"\n测试配置: {granularity} + {attention_type} + {model_size}")
                
                try:
                    config = create_enhanced_model_config(granularity, attention_type, model_size)
                    
                    model = EnhancedESA3DModel(
                        node_dim=config['node_dim'],
                        edge_dim=config['edge_dim'],
                        hidden_dim=config['hidden_dim'],
                        num_layers=config['num_layers'],
                        num_heads=config['num_heads'],
                        num_radial=config['num_radial'],
                        cutoff=config['cutoff'],
                        num_seeds=config['num_seeds'],
                        output_dim=1,
                        granularity=granularity,
                        attention_type=attention_type,
                        k_neighbors=config['k_neighbors'],
                        block_size=config['block_size'],
                        dropout=config['dropout'],
                    ).to(device)
                    
                    # 前向传播
                    with torch.no_grad():
                        output = model(data)
                    
                    print(f"  输出形状: {output.shape}")
                    print(f"  参数数量: {sum(p.numel() for p in model.parameters())}")
                    print(f"  测试通过")
                
                except Exception as e:
                    print(f"  错误: {e}")
