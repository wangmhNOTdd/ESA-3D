#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
简化版的ESA-3D模型，避免内存问题
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax


class SimpleESA3DModel(nn.Module):
    """简化版的ESA-3D模型"""
    
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
        num_layers: int = 6,
        output_dim: int = 1,
        dropout: float = 0.1,
        cutoff: float = 10.0,
        num_radial: int = 64,
    ):
        super().__init__()
        
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.cutoff = cutoff
        
        # 节点嵌入
        self.node_embedding = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # 边嵌入
        self.edge_embedding = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # 径向基函数
        self.rbf = RadialBasisFunction(num_radial, cutoff)
        
        # 消息传递层
        self.layers = nn.ModuleList([
            SimpleMessagePassing(hidden_dim, num_radial, dropout)
            for _ in range(num_layers)
        ])
        
        # 输出层
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        
    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        block_ids: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: [N, node_dim] 节点特征
            pos: [N, 3] 节点位置
            edge_index: [2, E] 边索引
            block_ids: [N] 节点的区块ID
            batch: [N] 批次索引
            edge_attr: [E, edge_dim] 边特征
        
        Returns:
            [batch_size, output_dim] 预测结果
        """
        # 节点嵌入
        h = self.node_embedding(x)
        
        # 边嵌入
        if edge_attr is not None:
            edge_emb = self.edge_embedding(edge_attr)
        else:
            edge_emb = None
        
        # 计算边距离和径向基函数
        row, col = edge_index
        edge_vec = pos[row] - pos[col]
        edge_dist = torch.norm(edge_vec, dim=-1)
        rbf_features = self.rbf(edge_dist)
        
        # 消息传递
        for layer in self.layers:
            h = layer(h, edge_index, rbf_features, edge_emb)
        
        # 全局池化
        if batch is None:
            batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)
        
        # 简单的平均池化
        batch_size = batch.max().item() + 1
        graph_features = torch.zeros(batch_size, self.hidden_dim, device=h.device)
        
        for i in range(batch_size):
            mask = (batch == i)
            if mask.any():
                graph_features[i] = h[mask].mean(dim=0)
        
        # 输出预测
        return self.output(graph_features)


class SimpleMessagePassing(MessagePassing):
    """简化的消息传递层"""
    
    def __init__(self, hidden_dim: int, num_radial: int, dropout: float = 0.1):
        super().__init__(aggr='add')
        
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        # 消息函数
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + num_radial, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # 更新函数
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # 边嵌入MLP
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        rbf_features: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: [N, hidden_dim] 节点特征
            edge_index: [2, E] 边索引
            rbf_features: [E, num_radial] 径向基函数特征
            edge_attr: [E, hidden_dim] 边特征
        
        Returns:
            [N, hidden_dim] 更新后的节点特征
        """
        # 保存残差连接
        residual = x
        
        # 消息传递
        out = self.propagate(edge_index, x=x, rbf_features=rbf_features, edge_attr=edge_attr)
        
        # 残差连接
        return residual + out
    
    def message(self, x_i, x_j, rbf_features, edge_attr):
        """构建消息"""
        # 基本消息：源节点 + 目标节点 + 径向基函数
        msg = torch.cat([x_i, x_j, rbf_features], dim=-1)
        
        # 通过MLP处理
        msg = self.message_mlp(msg)
        
        # 如果有边特征，融合边信息
        if edge_attr is not None:
            edge_msg = self.edge_mlp(edge_attr)
            msg = msg + edge_msg
        
        return msg
    
    def update(self, aggr_out, x):
        """更新节点特征"""
        # 聚合消息 + 原始特征
        combined = torch.cat([aggr_out, x], dim=-1)
        return self.update_mlp(combined)


class RadialBasisFunction(nn.Module):
    """径向基函数"""
    
    def __init__(self, num_radial: int, cutoff: float):
        super().__init__()
        
        self.num_radial = num_radial
        self.cutoff = cutoff
        
        # 可学习的中心和宽度
        self.centers = nn.Parameter(torch.linspace(0, cutoff, num_radial))
        self.widths = nn.Parameter(torch.ones(num_radial) * 0.5)
        
    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """
        计算径向基函数
        
        Args:
            distances: [E] 边距离
        
        Returns:
            [E, num_radial] 径向基函数特征
        """
        # 扩展维度用于广播
        distances = distances.unsqueeze(-1)  # [E, 1]
        centers = self.centers.unsqueeze(0)  # [1, num_radial]
        widths = self.widths.unsqueeze(0)    # [1, num_radial]
        
        # 高斯径向基函数
        rbf = torch.exp(-widths * (distances - centers) ** 2)
        
        # 截断函数
        cutoff_fn = 0.5 * (torch.cos(distances * torch.pi / self.cutoff) + 1.0)
        cutoff_fn = cutoff_fn * (distances < self.cutoff).float()
        
        return rbf * cutoff_fn
