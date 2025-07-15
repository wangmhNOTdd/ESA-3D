#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch
from torch_scatter import scatter_mean, scatter_sum
from typing import Optional, Tuple, List

from modules.equivariant_edge_attention import (
    ESA3DBlock,
    create_intra_block_mask,
    create_inter_block_mask,
)


class EdgeEmbedding(nn.Module):
    """边嵌入层，将节点特征和边特征合并为边的初始表示"""
    
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
        activation: nn.Module = nn.SiLU(),
    ):
        super().__init__()
        
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        
        # 节点特征到边特征的映射
        input_dim = node_dim * 2  # 连接两个节点的特征
        if edge_dim > 0:
            input_dim += edge_dim
            
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
            edge_index: [2, E] 边索引
            edge_attr: [E, edge_dim] 边属性(可选)
        
        Returns:
            edge_features: [E, hidden_dim] 边的不变特征
            edge_coords: [E, 3] 边的等变特征(相对位置向量)
        """
        row, col = edge_index
        
        # 构建边的不变特征
        edge_features = torch.cat([node_features[row], node_features[col]], dim=-1)
        
        if edge_attr is not None:
            edge_features = torch.cat([edge_features, edge_attr], dim=-1)
        
        edge_features = self.edge_mlp(edge_features)
        
        # 构建边的等变特征：相对位置向量
        edge_coords = node_coords[row] - node_coords[col]  # [E, 3]
        
        return edge_features, edge_coords


class AttentionPooling(nn.Module):
    """注意力池化模块，将边特征聚合为全局图表示"""
    
    def __init__(
        self,
        hidden_dim: int,
        num_seeds: int = 32,
        num_heads: int = 8,
        activation: nn.Module = nn.SiLU(),
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_seeds = num_seeds
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0
        
        # 可学习的种子向量
        self.seed_vectors = nn.Parameter(torch.randn(num_seeds, hidden_dim))
        
        # 多头注意力
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        
        # 最终投影
        self.final_mlp = nn.Sequential(
            nn.Linear(hidden_dim * num_seeds, hidden_dim),
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
            
            # 添加坐标的范数作为不变特征
            coord_norms = torch.norm(batch_edge_coords, dim=-1, keepdim=True)  # [E_b, 1]
            augmented_features = torch.cat([batch_edge_features, coord_norms], dim=-1)  # [E_b, hidden_dim+1]
            
            # 投影到原始维度
            augmented_features = F.linear(augmented_features, 
                                        torch.cat([torch.eye(self.hidden_dim, device=edge_features.device), 
                                                 torch.zeros(self.hidden_dim, 1, device=edge_features.device)], dim=-1))
            
            # 生成Query (种子向量), Key和Value (边特征)
            q = self.q_proj(self.seed_vectors).reshape(self.num_seeds, self.num_heads, self.head_dim)
            k = self.k_proj(augmented_features).reshape(augmented_features.shape[0], self.num_heads, self.head_dim)
            v = self.v_proj(augmented_features).reshape(augmented_features.shape[0], self.num_heads, self.head_dim)
            
            # 计算注意力
            attention_scores = torch.einsum('shd,ehd->she', q, k) / (self.head_dim ** 0.5)
            attention_weights = torch.softmax(attention_scores, dim=-1)  # [num_seeds, num_heads, E_b]
            
            # 聚合
            pooled = torch.einsum('she,ehd->shd', attention_weights, v)  # [num_seeds, num_heads, head_dim]
            pooled = pooled.reshape(self.num_seeds, self.hidden_dim)  # [num_seeds, hidden_dim]
            
            # 最终池化
            graph_repr = self.final_mlp(pooled.reshape(-1))  # [hidden_dim]
            graph_reprs.append(graph_repr)
        
        if len(graph_reprs) == 0:
            return torch.zeros(0, self.hidden_dim, device=edge_features.device)
        
        return torch.stack(graph_reprs, dim=0)  # [B, hidden_dim]


class ESA3DEncoder(nn.Module):
    """ESA-3D编码器：交错的等变边注意力"""
    
    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 6,
        num_heads: int = 8,
        num_radial: int = 64,
        cutoff: float = 10.0,
        activation: nn.Module = nn.SiLU(),
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 构建多层ESA-3D块
        self.layers = nn.ModuleList([
            ESA3DBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_radial=num_radial,
                cutoff=cutoff,
                activation=activation,
                dropout=dropout,
            ) for _ in range(num_layers)
        ])
        
    def forward(
        self,
        edge_features: torch.Tensor,  # [E, hidden_dim]
        edge_coords: torch.Tensor,    # [E, 3]
        edge_index: torch.Tensor,     # [2, E]
        block_ids: torch.Tensor,      # [N] 每个节点的区块ID
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            edge_features: [E, hidden_dim] 边的不变特征
            edge_coords: [E, 3] 边的等变特征
            edge_index: [2, E] 边的索引
            block_ids: [N] 每个节点的区块ID
        
        Returns:
            final_edge_features: [E, hidden_dim] 最终的边不变特征
            final_edge_coords: [E, 3] 最终的边等变特征
        """
        device = edge_features.device
        
        # 创建掩码
        intra_mask = create_intra_block_mask(edge_index, block_ids, device)
        inter_mask = create_inter_block_mask(edge_index, block_ids, device)
        
        # 逐层处理
        current_edge_features = edge_features
        current_edge_coords = edge_coords
        
        for layer in self.layers:
            current_edge_features, current_edge_coords = layer(
                current_edge_features,
                current_edge_coords,
                intra_mask,
                inter_mask,
                edge_index,
            )
        
        return current_edge_features, current_edge_coords


class ESA3DModel(nn.Module):
    """ESA-3D模型：基于边集合注意力的3D分子性质预测"""
    
    def __init__(
        self,
        node_dim: int,
        edge_dim: int = 0,
        hidden_dim: int = 128,
        num_layers: int = 6,
        num_heads: int = 8,
        num_radial: int = 64,
        cutoff: float = 10.0,
        num_seeds: int = 32,
        output_dim: int = 1,
        activation: nn.Module = nn.SiLU(),
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        
        # 边嵌入
        self.edge_embedding = EdgeEmbedding(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            activation=activation,
        )
        
        # ESA-3D编码器
        self.encoder = ESA3DEncoder(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            num_radial=num_radial,
            cutoff=cutoff,
            activation=activation,
            dropout=dropout,
        )
        
        # 注意力池化
        self.pooling = AttentionPooling(
            hidden_dim=hidden_dim,
            num_seeds=num_seeds,
            num_heads=num_heads,
            activation=activation,
        )
        
        # 输出层
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, hidden_dim // 2),
            activation,
            nn.Linear(hidden_dim // 2, output_dim),
        )
        
    def forward(
        self,
        node_features: torch.Tensor,  # [N, node_dim]
        node_coords: torch.Tensor,    # [N, 3]
        edge_index: torch.Tensor,     # [2, E]
        block_ids: torch.Tensor,      # [N] 每个节点的区块ID
        batch: Optional[torch.Tensor] = None,  # [N] 批次信息
        edge_attr: Optional[torch.Tensor] = None,  # [E, edge_dim]
    ) -> torch.Tensor:
        """
        Args:
            node_features: [N, node_dim] 节点特征
            node_coords: [N, 3] 节点坐标  
            edge_index: [2, E] 边索引
            block_ids: [N] 每个节点的区块ID
            batch: [N] 批次信息
            edge_attr: [E, edge_dim] 边属性
        
        Returns:
            output: [B, output_dim] 预测结果
        """
        # 边嵌入
        edge_features, edge_coords = self.edge_embedding(
            node_features, node_coords, edge_index, edge_attr
        )
        
        # ESA-3D编码
        final_edge_features, final_edge_coords = self.encoder(
            edge_features, edge_coords, edge_index, block_ids
        )
        
        # 创建边的批次信息
        if batch is not None:
            edge_batch = batch[edge_index[0]]  # 每条边属于哪个批次
        else:
            edge_batch = None
        
        # 注意力池化
        graph_repr = self.pooling(final_edge_features, final_edge_coords, edge_batch)
        
        # 输出预测
        output = self.output_mlp(graph_repr)
        
        return output


# 工具函数
def edge_to_node_aggregation(
    edge_features: torch.Tensor,  # [E, hidden_dim]
    edge_coords: torch.Tensor,    # [E, 3]
    edge_index: torch.Tensor,     # [2, E]
    num_nodes: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """将边特征聚合回节点特征"""
    
    # 不变特征聚合
    node_features = scatter_mean(edge_features, edge_index[0], dim=0, dim_size=num_nodes)
    
    # 等变特征聚合
    node_coords = scatter_mean(edge_coords, edge_index[0], dim=0, dim_size=num_nodes)
    
    return node_features, node_coords


def create_line_graph(edge_index: torch.Tensor) -> torch.Tensor:
    """从原始图创建线图的边索引"""
    
    E = edge_index.shape[1]
    line_edges = []
    
    for i in range(E):
        for j in range(i + 1, E):
            edge_i = edge_index[:, i]
            edge_j = edge_index[:, j]
            
            # 如果两条边共享一个节点，则在线图中连接
            if edge_i[0] == edge_j[0] or edge_i[0] == edge_j[1] or edge_i[1] == edge_j[0] or edge_i[1] == edge_j[1]:
                line_edges.append([i, j])
                line_edges.append([j, i])  # 无向图
    
    if len(line_edges) == 0:
        return torch.zeros(2, 0, dtype=torch.long, device=edge_index.device)
    
    return torch.tensor(line_edges, dtype=torch.long, device=edge_index.device).t()
